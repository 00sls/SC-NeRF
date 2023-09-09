import sys, os, imageio, lpips

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

# def decode_batch(batch):
#     rays = batch['rays']  # (B, 8)
#     rgbs = batch['rgbs']  # (B, 3)
#     return rays, rgbs


def unpreprocess(data, shape=(1, 1, 3, 1, 1)):
    # to unnormalize image for visualization
    # data N V C H W
    device = data.device
    mean = torch.tensor([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]).view(*shape).to(device)
    std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)

    return (data - mean) / std

def decode_batch(batch, idx=list(torch.arange(4))):

    data_mvs = sub_selete_data(batch, device, idx, filtKey=[])
    pose_ref = {'w2cs': data_mvs['w2cs'].squeeze(), 'intrinsics': data_mvs['intrinsics'].squeeze(),
                'c2ws': data_mvs['c2ws'].squeeze(), 'near_fars': data_mvs['near_fars'].squeeze()}

    return data_mvs, pose_ref

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
mse2psnr = lambda x: -10. * np.log(x) / np.log(10.)

max_psnr = 0
# ckpts = os.listdir("/data/best_1023/mvsnerf_left_right/ckpts")
# ckpt_remove = []
# for i in ckpt_remove:
#     ckpts.remove(i)
ckpts = ['143999.tar']
for ckpt in ckpts:
    dir_ckpt = ckpt.split('.')[0]
    
    save_dir = f'/data/best_1023/mvsnerf_left_right/results/small_diff/nerf_space_{dir_ckpt}'  # small # media # diff
    os.makedirs(save_dir, exist_ok=True)
    
    f_txt = open(save_dir + '/' + 'out.txt', "w")
    f_txt.write(f'test_{save_dir} \n')
    psnr_all, ssim_all, LPIPS_vgg_all = [], [], []
    depth_acc = {}
    eval_metric = [0.1, 0.05, 0.01]
    depth_acc[f'abs_err'], depth_acc[f'acc_l_{eval_metric[0]}'], depth_acc[f'acc_l_{eval_metric[1]}'], depth_acc[
        f'acc_l_{eval_metric[2]}'] = {}, {}, {}, {}

    cmd = f'--datadir /dataset/spaces_dataset  \
        --dataset_name spaces  \
        --net_type v0 --ckpt ./best/{ckpt}'  # 这里需要修改
    args = config_parser(cmd.split())
    dataset_val = dataset_dict[args.dataset_name](dataset_path=args.datadir, is_val=True, layout='small_4_1',  # small_4_1 # medium_4_1 # difficult_4_1
                                                n_planes=args.N_samples, im_w=800, im_h=480)


    psnr, ssim, LPIPS_vgg = [], [], []


    args.use_viewdirs = True

    args.N_samples = 128
    args.feat_dim = 8 + 12  # 修改

        # create models

    render_kwargs_train, render_kwargs_test, start, grad_vars = create_nerf_mvs(args, use_mvs=True,
                                                                                dir_embedder=False,
                                                                                pts_embedder=True)
    filter_keys(render_kwargs_train)

    MVSNet = render_kwargs_train['network_mvs']
    render_kwargs_train.pop('network_mvs')
    # correct_mode = render_kwargs_train['Correct_model']
    # print(MVSNet)
    datadir = args.datadir
    datatype = 'val'
    pad = 24
    args.chunk = 5120
    args.netchunk = 5120
    print('============> rendering dataset <===================')

    save_as_image = True
    MVSNet.train()
    MVSNet = MVSNet.cuda()
    # correct_mode.train()
    # correct_mode = correct_mode.cuda()
        
    val_dataset = DataLoader(dataset_val,
            shuffle=False,
            num_workers=0,
            batch_size=1,
            pin_memory=True)

    with torch.no_grad():

        try:
            tqdm._instances.clear()
        except Exception:
            pass

        for i, batch in enumerate(val_dataset):
            
            data_mvs, pose_ref = decode_batch(batch)
            # print(batch['near_fars'])
            imges, proj_mats = data_mvs['images'], data_mvs['proj_mats']
            near_fars = pose_ref['near_fars']
            # print("near_fars",near_fars)
            depths_h = data_mvs['depths_h']
            H, W = imges.shape[-2:]
            H, W = int(H), int(W)

            world_to_ref = pose_ref['w2cs'][0]
            tgt_to_world, intrinsic = pose_ref['c2ws'][-1], pose_ref['intrinsics'][-1]
            volume_feature, img_feat, depth_values = MVSNet(imges[:, :3], proj_mats[:, :3], near_fars[0],
                                                            pad=args.pad)
            # volume_feature = torch.zeros((1, 8, 128, 176, 208)).float()
            imgs = unpreprocess(imges)

            rgb_rays, depth_rays_preds = [], []
            for chunk_idx in range(H * W // args.chunk + int(H * W % args.chunk > 0)):
                rays_pts, rays_dir, rays_NDC, depth_candidates, rays_o, ndc_parameters, ndc_3d = \
                    build_rays_test(H, W, tgt_to_world, world_to_ref, intrinsic, near_fars, near_fars[-1], args.N_samples, pad=args.pad, chunk=args.chunk, idx=chunk_idx)
                feat_sampled = compute_projections(imges, rays_pts, pose_ref, img_feat, H, W)
                
                rgb, disp, acc, depth_map_fn, depth_pred, density_ray, ret = rendering(args, feat_sampled, ndc_3d, pose_ref, rays_pts, rays_NDC, depth_candidates, rays_o, rays_dir,
                                                       volume_feature, imgs[:, :-1], img_feat=None,  **render_kwargs_train)

                rgb, depth_pred = torch.clamp(rgb.cpu(), 0, 1.0).numpy(), depth_pred.cpu().numpy()
                rgb_rays.append(rgb)
                depth_rays_preds.append(depth_pred)
            depth_rays_preds = np.concatenate(depth_rays_preds).reshape(H, W)
            rgb_rays = np.concatenate(rgb_rays).reshape(H, W, 3)
            img1 = imgs[0, 0].permute(1, 2, 0).cpu().numpy()
            img2 = imgs[0, 1].permute(1, 2, 0).cpu().numpy()
            img3 = imgs[0, 2].permute(1, 2, 0).cpu().numpy()
            img = imgs[0, 3].permute(1, 2, 0).cpu().numpy()
            img_vis = np.concatenate((img1*255,img2*255,img3*255,img*255,rgb_rays*255),axis=1)
            if save_as_image:
                print(f'{save_dir}/scan_{i:03d}.png')
                imageio.imwrite(f'{save_dir}/scan_{i:03d}.png', img_vis.astype('uint8'))
            else:
                rgbs.append(img_vis.astype('uint8'))
            # print(rgb_rays.shape)
            # print(img.shape)
            mpsnr = mse2psnr(np.mean((rgb_rays - img) ** 2))
            mssin = structural_similarity(rgb_rays, img, multichannel=True)
            
            psnr.append(mpsnr)
            ssim.append(mssin)
            img_tensor = torch.from_numpy(rgb_rays)[None].permute(0, 3, 1,
                                        2).float() * 2 - 1.0  # image should be RGB, IMPORTANT: normalized to [-1,1]
            img_gt_tensor = torch.from_numpy(img)[None].permute(0, 3, 1, 2).float() * 2 - 1.0
            lpips = loss_fn_vgg(img_tensor, img_gt_tensor).item()
            LPIPS_vgg.append(lpips)
            print(mpsnr, mssin, lpips)
            f_txt.write(f'=====> psnr {mpsnr} ssim: {mssin} lpips: {lpips} \n')
    print(f'=====> all mean psnr {np.mean(psnr)} ssim: {np.mean(ssim)} lpips: {np.mean(LPIPS_vgg)}')
    f_txt.write(f'=====> all mean psnr {np.mean(psnr)} ssim: {np.mean(ssim)} lpips: {np.mean(LPIPS_vgg)} \n')
    f_txt.close()
    mean_psnr = np.mean(psnr_all)
    # if max_psnr < mean_psnr:
    #     max_psnr = mean_psnr
    #     max_psnr_skpt = ckpt
    #     print(f'max psnr now {max_psnr}')
# print(f'max spnr ckpt is ' + max_psnr_skpt)  
