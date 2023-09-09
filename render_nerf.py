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


loss_fn_vgg = lpips.LPIPS(net='vgg') 
mse2psnr = lambda x : -10. * np.log(x) / np.log(10.)

ckpt = os.listdir("./best_128")
list_ckpt = []
for item in ckpt:
    if item[-3:] == "tar":
        list_ckpt.append(item)
ckpt_remove = []


for i in ckpt_remove:
    try:
        list_ckpt.remove(i)
    except:
        pass
print(list_ckpt)

for ckpt in list_ckpt:
    psnr_all,ssim_all,LPIPS_vgg_all = [],[],[]
    for i_scene, scene in enumerate(['ship','mic','chair','lego','drums','ficus','materials','hotdog']):#
        psnr,ssim,LPIPS_vgg = [],[],[]
        cmd = f'--datadir /dataset/mvsnerf/nerf_synthetic/{scene}  \
        --dataset_name blender --white_bkgd \
        --net_type v0 --ckpt ./best_128/{ckpt}'

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
        ckpt_name = ckpt.split('.')[0]
        save_dir = f'results/render_nerf_{ckpt_name}'
        os.makedirs(save_dir, exist_ok=True)
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
            
                # find nearest image idx from training views
                positions = dataset_train.poses[:,:3,3]
                dis = np.sum(np.abs(positions - dataset_val.poses[[i],:3,3]), axis=-1)
                pair_idx = np.argsort(dis)[:3]
                pair_idx = [dataset_train.img_idx[item] for item in pair_idx]
                
                imges_source, proj_mats, near_far_source, pose_source = dataset_train.read_source_views(pair_idx=pair_idx,device=device)
                # print("near_far_source", near_far_source)
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
                    intrinsic_ref[:2] *= args.imgScale_test/args.imgScale_train
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
                depth_rays_preds, _ = visualize_depth_numpy(depth_rays_preds, near_far_source)
                
                rgb_rays = np.concatenate(rgb_rays).reshape(H, W, 3)
                img_vis = np.concatenate((img*255,rgb_rays*255,depth_rays_preds),axis=1)
                
    #             img_vis = np.concatenate((torch.cat(torch.split(imgs_source*255, [1,1,1], dim=1),-1).squeeze().permute(1,2,0).cpu().numpy(),img_vis),axis=1)
                
                if save_as_image:
                    imageio.imwrite(f'{save_dir}/{scene}_{val_idx[i]:03d}.png', img_vis.astype('uint8'))
                else:
                    rgbs.append(img_vis.astype('uint8'))
                    
                # quantity
                # center crop 0.8 ratio
                H_crop, W_crop = np.array(rgb_rays.shape[:2])//10
                img = img[H_crop:-H_crop,W_crop:-W_crop]
                rgb_rays = rgb_rays[H_crop:-H_crop,W_crop:-W_crop]
                
                psnr.append( mse2psnr(np.mean((rgb_rays-img)**2)))
                ssim.append( structural_similarity(rgb_rays, img, multichannel=True))
                
                img_tensor = torch.from_numpy(rgb_rays)[None].permute(0,3,1,2).float()*2-1.0 # image should be RGB, IMPORTANT: normalized to [-1,1]
                img_gt_tensor = torch.from_numpy(img)[None].permute(0,3,1,2).float()*2-1.0
                LPIPS_vgg.append( loss_fn_vgg(img_tensor, img_gt_tensor).item())

            print(f'=====> scene: {scene} mean psnr {np.mean(psnr)} ssim: {np.mean(ssim)} lpips: {np.mean(LPIPS_vgg)}')   
            psnr_all.append(psnr);ssim_all.append(ssim);LPIPS_vgg_all.append(LPIPS_vgg)

        if not save_as_image:
            imageio.mimwrite(f'{save_dir}/{scene}_spiral.mp4', np.stack(rgbs), fps=20, quality=10)

    print(f'=====> all mean psnr {np.mean(psnr_all)} ssim: {np.mean(ssim_all)} lpips: {np.mean(LPIPS_vgg_all)}') 