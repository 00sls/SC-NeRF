import sys,os,imageio,lpips
root = '/data/best_1023/mvsnerf_left_right'
os.chdir(root)
sys.path.append(root)

from opt import config_parser
from data import dataset_dict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# models
from models import *
from renderer import *
from data.ray_utils import get_rays

from tqdm import tqdm

from skimage.metrics import structural_similarity

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, loggers


from data.ray_utils import ray_marcher

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )


def decode_batch(batch):
    rays = batch['rays']  # (B, 8)
    rgbs = batch['rgbs']  # (B, 3)
    return rays, rgbs

def unpreprocess(data, shape=(1,1,3,1,1)):
    # to unnormalize image for visualization
    # data N V C H W
    device = data.device
    mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
    std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

    return (data - mean) / std

def read_depth(filename):
    depth_h = np.array(read_pfm(filename)[0], dtype=np.float32) # (800, 800)
    depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_NEAREST)  # (600, 800)
    depth_h = depth_h[44:556, 80:720]  # (512, 640)
#     depth = cv2.resize(depth_h, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_NEAREST)#!!!!!!!!!!!!!!!!!!!!!!!!!
    mask = depth>0
    return depth_h,mask

def inbound_(pixel_locations, h, w):
    '''
    check if the pixel locations are in valid range
    :param pixel_locations: [..., 2]
    :param h: height
    :param w: weight
    :return: mask, bool, [...]
    '''
    return (pixel_locations[..., 0] <= w - 1.) & \
            (pixel_locations[..., 0] >= 0) & \
            (pixel_locations[..., 1] <= h - 1.) &\
            (pixel_locations[..., 1] >= 0)

def compute_projections(imges, xyz, pose_ref, feature, h, w):
    '''
    project 3D points into cameras
    :param xyz: [..., 3]
    :param train_cameras: [n_views, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
    :return: pixel locations [..., 2], mask [...]
    '''
    # print(volume.shape)
    
    # print(xyz.shape)
    original_shape = xyz.shape[:2]
    xyz = xyz.reshape(-1, 3)
    train_poses = pose_ref['c2ws'][:3]
    # print(train_poses.shape)
    train_intrinsics = torch.cat([torch.eye(4).to(device).unsqueeze(0), torch.eye(4).to(device).unsqueeze(0), torch.eye(4).to(device).unsqueeze(0)],dim=0)
    # print(train_intrinsics.shape)
    # print(pose_ref['intrinsics'][:3].shape)
    intrinsics = pose_ref['intrinsics'][:3].clone()
    intrinsics[:,:2] = intrinsics[:, :2]
    train_intrinsics[:,:3,:3] = intrinsics
    # print(train_intrinsics.shape)
    num_views = 3
    
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]
    # print(xyz_h.shape)
    projections = train_intrinsics.bmm(torch.inverse(train_poses)) \
        .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))  # [n_views, 4, n_points]
    projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
    pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
    pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
    mask = projections[..., 2] > 0   # a point is invalid if behind the camera
    pixel_locations = pixel_locations.reshape((num_views, ) + original_shape + (2, ))
    mask_in_front = mask.reshape((num_views, ) + original_shape)
    # print(torch.min(pixel_locations), torch.max(pixel_locations))
    # print(mask_in_front.shape)
    # torch.Size([3, 1024, 128, 2])
    # torch.Size([3, 1024, 128])
    resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, :]
    normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
    # print(torch.min(normalized_pixel_locations), torch.max(normalized_pixel_locations))
    # print(imges.shape)
    rgbs_sampled = F.grid_sample(imges[0, 0:3], normalized_pixel_locations, align_corners=True)
    rgbs_sampled = rgbs_sampled.permute(0, 2, 3, 1).contiguous()     
    
    feat_sampled = F.grid_sample(feature[0], normalized_pixel_locations, align_corners=True)
    feat_sampled = feat_sampled.permute(0, 2, 3, 1).contiguous()     
    rgb_feat_sampled = torch.cat([rgbs_sampled, feat_sampled], dim=-1)   # [n_rays, n_samples, n_views, d+3]
    # print(rgb_feat_sampled.shape)
    inbound = inbound_(pixel_locations, h, w)
    mask = (inbound * mask_in_front).float().permute(1, 2, 0)[..., None].contiguous()  
    mask = mask.permute(2, 0, 1, 3).contiguous()  
    feat_sampled = rgb_feat_sampled*mask
    # print(feat_sampled.shape)
    # print(mask.shape)  # torch.Size([1024, 128, 3, 1])
    # print(torch.min(mask), torch.min(mask), mask[0, 0])
    
    return feat_sampled

def acc_threshold(abs_err, threshold):
    """
    computes the percentage of pixels whose depth error is less than @threshold
    """
    acc_mask = abs_err < threshold
    return  acc_mask.astype('float') if type(abs_err) is np.ndarray else acc_mask.float()

loss_fn_vgg = lpips.LPIPS(net='vgg') 
mse2psnr = lambda x : -10. * np.log(x) / np.log(10.)
expname = 'mvs-nerf-dtu_step'

ckpts = os.listdir(f"./runs_new/{expname}/ckpts")
max_psnr = 0
max_psnr_skpt = ' '
ckpt_remove = []


# ['10499.tar', '10999.tar', '11499.tar', '11999.tar', '12499.tar', '12999.tar', '13499.tar', '13999.tar', '14499.tar', '1499.tar', '14999.tar', '15499.tar', '15999.tar', '16499.tar', '16999.tar', '17499.tar', '17999.tar', '18499.tar', '18999.tar', '19499.tar', '1999.tar', '19999.tar', '20499.tar', '20999.tar', '21499.tar', '21999.tar', '22499.tar', '22999.tar', '23499.tar', '23999.tar', '24499.tar', '2499.tar', '24999.tar', '25499.tar', '25999.tar', '26499.tar', '26999.tar', '27499.tar', '27999.tar', '28499.tar', '2999.tar', '3499.tar', '3999.tar', '4499.tar', '499.tar', '4999.tar', '5499.tar', '5999.tar', '6499.tar', '6999.tar', '7499.tar', '7999.tar', '8499.tar', '8999.tar', '9499.tar', '999.tar', '9999.tar']

for i in ckpt_remove:
    try:
        ckpts.remove(i)
    except:
        pass
# ckpts = ['71999.tar', '72999.tar', '73999.tar', '74999.tar', '75999.tar', '76999.tar', '77999.tar']
ckpts.sort()
ckpts_val = ['499.tar']
# for i in ckpts:
#     if int(i.split('.')[0]) > 80000:
#         ckpts_val.append(i)
print(ckpts_val)
psnr_out, ssim_out, lpip_out = [], [], []
for ckpt in ckpts_val:
    dir_ckpt = ckpt.split('.')[0]
    save_dir = f'/data/best_1023/mvsnerf_left_right/results/{expname}/nerf_dtu_{dir_ckpt}'
    os.makedirs(save_dir, exist_ok=True)
    f_txt = open(save_dir + '/' + 'out.txt', "w")
    psnr_all,ssim_all,LPIPS_vgg_all = [],[],[]
    depth_acc = {}
    eval_metric = [0.1,0.05,0.01]
    depth_acc[f'abs_err'],depth_acc[f'acc_l_{eval_metric[0]}'],depth_acc[f'acc_l_{eval_metric[1]}'],depth_acc[f'acc_l_{eval_metric[2]}'] = {},{},{},{}
    for i_scene, scene in enumerate([1,8,21,103,114]):#,8,21,103,114
        psnr,ssim,LPIPS_vgg = [],[],[]
        cmd = f'--datadir /dataset/mvsnerf/mvs_training/dtu/scan{scene}  \
        --dataset_name dtu_ft  \
        --net_type v0 --ckpt /data/best_1023/mvsnerf_left_right/runs_new/{expname}/ckpts/{ckpt}'

        args = config_parser(cmd.split())
        args.use_viewdirs = True

        args.N_samples = 128
        args.feat_dim =  8+12

        # create models
        if 0==i_scene:
            render_kwargs_train, render_kwargs_test, start, grad_vars = create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True)
            filter_keys(render_kwargs_train)

            MVSNet = render_kwargs_train['network_mvs']
            render_kwargs_train.pop('network_mvs')
            # rgb_model = render_kwargs_train['rgb_model']

        datadir = args.datadir
        datatype = 'train'
        pad = 24
        args.chunk = 5120
        args.netchunk = 5120

        print('============> rendering dataset <===================')
        dataset_train = dataset_dict[args.dataset_name](args, split='train')
        dataset_val = dataset_dict[args.dataset_name](args, split='val')
        val_idx = dataset_val.img_idx
        
        save_as_image = True
        
        
        MVSNet.train()
        MVSNet = MVSNet.cuda()

        with torch.no_grad():

            try:
                tqdm._instances.clear() 
            except Exception:     
                pass
            
            for i, batch in enumerate(tqdm(dataset_val)):
                torch.cuda.empty_cache()
                
                rays, img = decode_batch(batch)
                rays = rays.squeeze().to(device)  # (H*W, 3)
                img = img.squeeze().cpu().numpy()  # (H, W, 3)
                depth = batch['depth'].squeeze().numpy()  # (H, W)
                del batch
                # find nearest image idx from training views
                positions = dataset_train.poses[:,:3,3]
                dis = np.sum(np.abs(positions - dataset_val.poses[[i],:3,3]), axis=-1)
                pair_idx = np.argsort(dis)[:3]
                pair_idx = [dataset_train.img_idx[item] for item in pair_idx]
                
                imges_source, proj_mats, near_far_source, pose_source = dataset_train.read_source_views(pair_idx=pair_idx,device=device)
                volume_feature, img_feat, _ = MVSNet(imges_source, proj_mats, near_far_source, pad=pad)
                imgs_source = unpreprocess(imges_source)
            
                N_rays_all = rays.shape[0]
                rgb_rays, depth_rays_preds = [],[]
                for chunk_idx in range(N_rays_all//args.chunk + int(N_rays_all%args.chunk>0)):

                    xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
                                                        N_samples=args.N_samples)

                    # Converting world coordinate to ndc coordinate
                    H, W = img.shape[:2]
                    H, W = int(H), int(W)
                    inv_scale = torch.tensor([W - 1, H - 1]).to(device)
                    w2c_ref, intrinsic_ref = pose_source['w2cs'][0], pose_source['intrinsics'][0].clone()
                    xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                                near=near_far_source[0], far=near_far_source[1], pad=pad*args.imgScale_test)
                    ndc_3d = get_ndc_coordinate(None, intrinsic_ref, xyz_coarse_sampled, inv_scale,
                                                near=near_far_source[0], far=near_far_source[1], pad=pad*args.imgScale_test)

                    feat_sampled = compute_projections(imges_source, xyz_coarse_sampled, pose_source, img_feat, H, W)
                    # rendering
                    rgb, disp, acc, depth_map_fn, depth_pred, alpha, extras = rendering(args,feat_sampled,ndc_3d, pose_source, xyz_coarse_sampled,
                                                                        xyz_NDC, z_vals, rays_o, rays_d,
                                                                        volume_feature,imgs_source, **render_kwargs_train)
        
                    
                    rgb, depth_pred = torch.clamp(rgb.cpu(),0,1.0).numpy(), depth_pred.cpu().numpy()
                    rgb_rays.append(rgb)
                    depth_rays_preds.append(depth_pred)

                
                depth_rays_preds = np.concatenate(depth_rays_preds).reshape(H, W)

                depth_gt, _ =  read_depth(f'/dataset/mvsnerf/mvs_training/dtu/Depths/scan{scene}/depth_map_{val_idx[i]:04d}.pfm')
            
                mask_gt = depth_gt>0
                abs_err = abs_error(depth_rays_preds, depth_gt/200, mask_gt)

                eval_metric = [0.01,0.05, 0.1]
                depth_acc[f'abs_err'][f'{scene}'] = np.mean(abs_err)
                depth_acc[f'acc_l_{eval_metric[0]}'][f'{scene}'] = acc_threshold(abs_err,eval_metric[0]).mean()
                depth_acc[f'acc_l_{eval_metric[1]}'][f'{scene}'] = acc_threshold(abs_err,eval_metric[1]).mean()
                depth_acc[f'acc_l_{eval_metric[2]}'][f'{scene}'] = acc_threshold(abs_err,eval_metric[2]).mean()

                
                depth_rays_preds, _ = visualize_depth_numpy(depth_rays_preds, near_far_source)
                depth_fn, _ = visualize_depth_numpy(depth_gt, near_far_source)
                
                rgb_rays = np.concatenate(rgb_rays).reshape(H, W, 3)
                img_vis = np.concatenate((img*255,rgb_rays*255,depth_rays_preds, depth_fn),axis=1)
                
                if save_as_image:
                    imageio.imwrite(f'{save_dir}/scan{scene}_{val_idx[i]:03d}.png', img_vis.astype('uint8'))
                else:
                    rgbs.append(img_vis.astype('uint8'))
                    
                # quantity
                # mask background since they are outside the far boundle
                mask = depth==0
                imageio.imwrite(f'{save_dir}/scan{scene}_{val_idx[i]:03d}_mask.png', mask.astype('uint8')*255)
                rgb_rays[mask],img[mask] = 0.0,0.0
                psnr.append( mse2psnr(np.mean((rgb_rays[~mask]-img[~mask])**2)))
                ssim.append( structural_similarity(rgb_rays, img, multichannel=True))
                
                img_tensor = torch.from_numpy(rgb_rays)[None].permute(0,3,1,2).float()*2-1.0 # image should be RGB, IMPORTANT: normalized to [-1,1]
                img_gt_tensor = torch.from_numpy(img)[None].permute(0,3,1,2).float()*2-1.0
                LPIPS_vgg.append( loss_fn_vgg(img_tensor, img_gt_tensor).item())

            print(f'=====> scene: {scene} mean psnr {np.mean(psnr)} ssim: {np.mean(ssim)} lpips: {np.mean(LPIPS_vgg)}')   
            f_txt.write('\n' + f'=====> scene: {scene} mean psnr {np.mean(psnr)} ssim: {np.mean(ssim)} lpips: {np.mean(LPIPS_vgg)}')
            psnr_all.append(psnr);ssim_all.append(ssim);LPIPS_vgg_all.append(LPIPS_vgg)
            

        if not save_as_image:
            imageio.mimwrite(f'{save_dir}/{scene}_spiral.mp4', np.stack(rgbs), fps=20, quality=10)

    a = np.mean(list(depth_acc['abs_err'].values()))
    b = np.mean(list(depth_acc[f'acc_l_{eval_metric[0]}'].values()))
    c = np.mean(list(depth_acc[f'acc_l_{eval_metric[1]}'].values()))
    d = np.mean(list(depth_acc[f'acc_l_{eval_metric[2]}'].values()))
    print(f'============> abs_err: {a} <=================')
    f_txt.write('\n' + f'============> abs_err: {a} <=================')
    print(f'============> acc_l_{eval_metric[0]}: {b} <=================')
    f_txt.write('\n' + f'============> acc_l_{eval_metric[0]}: {b} <=================')
    print(f'============> acc_l_{eval_metric[1]}: {c} <=================')
    f_txt.write('\n' + f'============> acc_l_{eval_metric[1]}: {c} <=================')
    print(f'============> acc_l_{eval_metric[2]}: {d} <=================')
    f_txt.write('\n' + f'============> acc_l_{eval_metric[2]}: {d} <=================')
    print(f'=====> all mean psnr {np.mean(psnr_all)} ssim: {np.mean(ssim_all)} lpips: {np.mean(LPIPS_vgg_all)}') 
    f_txt.write('\n' + f'=====> all mean psnr {np.mean(psnr_all)} ssim: {np.mean(ssim_all)} lpips: {np.mean(LPIPS_vgg_all)}')
    f_txt.close()
    psnr_out.append(np.mean(psnr_all))
    ssim_out.append(np.mean(ssim_all))
    lpip_out.append(np.mean(LPIPS_vgg_all))
    mean_psnr = np.mean(psnr_all)
    if mean_psnr > 26.63 and np.mean(LPIPS_vgg_all) < 0.1685:
        print(f"cp ./runs_new/{expname}/ckpts/{ckpt} ./best/")  # mvsnerf_left_right/runs_new/{expname}/ckpts/499.tar
    if max_psnr < mean_psnr:
        max_psnr = mean_psnr
        max_psnr_skpt = ckpt
        print(f'max psnr now {max_psnr}')
print(f'max spnr ckpt is ' + max_psnr_skpt)
print(ckpts)
print(psnr_out)
print(ssim_out)
print(lpip_out) 