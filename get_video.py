import sys,os,imageio
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
from scipy.spatial.transform import Rotation as R

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


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

def poses_avg(poses):

    center = poses[:, :3, 3].mean(0)
    print('center in poses_avg', center)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    
    return c2w

def render_path_imgs(c2ws_all, focal):
    T = c2ws_all[...,3]

    return render_poses

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * N_rots, N+1)[:-1]:
    # for theta in np.linspace(0., 5, N+1)[:-1]:
        # spiral
        # c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)

        # 关于使用平均姿态的相关问题
        # 从训练集推算出来的平均姿态方向基本平行于z轴，因为训练集中大多数图片是正面的，
        # 但是存在一个问题，将z和平均姿态相乘后得到的方向也基本上和z平行，所以无论怎么调整看起来都是平行的，
        # 别用平均姿态看其他位置的照片，直接用世界坐标系即可！！！！
        # 但是需要用别的姿态大致估计一下位置参数
        c = np.array([(np.cos(theta)*theta)/10, (-np.sin(theta)*theta)/10, -0.1]) 

        # 这个是因为作者在读取并规范化相机姿态的时候作了poses*blender2opencv，转换了坐标系，
        # 我用的数据无需转换，但是这里加个负号就解决了，目前不影响什么，记住就行
        z = -(normalize(c - np.array([0,0,-focal])))
        print("c", c)
        print("z", z)
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def get_spiral(c2ws_all, near_far, rads_scale=0.5, N_views=120):

    # center pose
    c2w = poses_avg(c2ws_all)
    print('poses_avg', c2w)
    
    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    close_depth, inf_depth = near_far
    print('near and far bounds', close_depth, inf_depth)
    dt = .75
    mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
    focal = mean_dz
    print(focal)

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = close_depth * .2
    tt = c2ws_all[:,:3,3] - c2w[:3,3][None]
    rads = np.percentile(np.abs(tt), 70, 0)*rads_scale
    print("rads",rads)
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=.5, N=N_views)
    return np.stack(render_poses)

def position2angle(position, N_views=16, N_rots = 2):
    ''' nx3 '''
    position = normalize(position)
    theta = np.arccos(position[:,2])/np.pi*180
    phi = np.arctan2(position[:,1],position[:,0])/np.pi*180
    return [theta,phi]

def pose_spherical_nerf(euler, radius=0.01):
    c2ws_render = np.eye(4)
    c2ws_render[:3,:3] =  R.from_euler('xyz', euler, degrees=True).as_matrix()
    # 保留旋转矩阵的最后一列再乘个系数就能当作位置？
    c2ws_render[:3,3]  = c2ws_render[:3,:3] @ np.array([0.0,0.0,-radius])
    return c2ws_render

def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.
    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -0.9 * t],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ])

        rot_phi = lambda phi: np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ])

        rot_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi / 5, radius)]  # 36 degree view downwards
    return np.stack(spheric_poses, 0)

def nerf_video_path(c2ws, theta_range=10,phi_range=20,N_views=120):
    c2ws = torch.tensor(c2ws)
    mean_position = torch.mean(c2ws[:,:3, 3],dim=0).reshape(1,3).cpu().numpy()
    rotvec = []
    for i in range(c2ws.shape[0]):
        r = R.from_matrix(c2ws[i, :3, :3])
        euler_ange = r.as_euler('xyz', degrees=True).reshape(1, 3)
        if i:
            mask = np.abs(euler_ange - rotvec[0])>180
            euler_ange[mask] += 360.0
        rotvec.append(euler_ange)
    # 采用欧拉角做平均的方法求旋转矩阵的平均
    rotvec = np.mean(np.stack(rotvec), axis=0)
#     render_poses = [pose_spherical_nerf(rotvec)]
    render_poses = [pose_spherical_nerf(rotvec+np.array([angle,0.0,-phi_range])) for angle in np.linspace(-theta_range,theta_range,N_views//4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec+np.array([theta_range,0.0,angle])) for angle in np.linspace(-phi_range,phi_range,N_views//4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec+np.array([angle,0.0,phi_range])) for angle in np.linspace(theta_range,-theta_range,N_views//4, endpoint=False)]
    render_poses += [pose_spherical_nerf(rotvec+np.array([-theta_range,0.0,angle])) for angle in np.linspace(phi_range,-phi_range,N_views//4, endpoint=False)]
    # render_poses = torch.from_numpy(np.stack(render_poses)).float().to(device)
    return render_poses



# llff
# for i_scene, scene in enumerate(['horns','flower','orchids', 'room','leaves','fern','trex','fortress']):#
#     # add --use_color_volume if the ckpts are fintuned with this flag
#     cmd = f'--datadir /dataset/mvsnerf/nerf_llff_data/{scene}  \
#      --dataset_name llff --imgScale_test {1.0}  --netwidth 128 --net_type v0 '# Please check whether finetuning setting is same with rendering setting, especially on use_color_volume, pad, and use_disp

#     is_finetued = True # set False if rendering without finetuning
#     if is_finetued:
#         cmd += f'--ckpt ./runs_fine_tuning/{scene}-ft/ckpts/latest.tar'
#     else:
#         cmd += '--ckpt ./ckpts/mvsnerf-v0.tar'
        
#     args = config_parser(cmd.split())
#     args.use_viewdirs = True

#     args.N_samples = 128
#     args.feat_dim =  8+3*4
# #     args.use_color_volume = False if not is_finetued else args.use_color_volume

#     # create models
#     if i_scene==0 or is_finetued:
#         render_kwargs_train, render_kwargs_test, start, grad_vars = create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True)
#         filter_keys(render_kwargs_train)

#         MVSNet = render_kwargs_train['network_mvs']
#         render_kwargs_train.pop('network_mvs')


#     datadir = args.datadir
#     datatype = 'val'
#     pad = 24 #the padding value should be same as your finetuning ckpt
#     args.chunk = 5120


#     dataset = dataset_dict[args.dataset_name](args, split=datatype)
#     val_idx = dataset.img_idx
    

#     save_dir = f'/data/best_1023/mvsnerf_head_eca_drop/results/videos'
#     os.makedirs(save_dir, exist_ok=True)
#     MVSNet.train()
#     MVSNet = MVSNet.cuda()
    
#     with torch.no_grad():

#         c2ws_all = dataset.poses

#         if is_finetued:   
#             # large baselien
#             imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(device=device)
#             volume_feature = torch.load(args.ckpt)['volume']['feat_volume']
#             volume_feature = RefVolume(volume_feature.detach()).cuda()
            
#             pad *= args.imgScale_test
#             w2cs, c2ws = pose_source['w2cs'], pose_source['c2ws']
#             pair_idx = torch.load('configs/pairs.th')[f'{scene}_train']
#             # 为啥这个地方只用训练集的相机姿态？
#             c2ws_render = get_spiral(c2ws_all[pair_idx], near_far_source, rads_scale = 10, N_views=30)# you can enlarge the rads_scale if you want to render larger baseline
#             # c2ws_render = nerf_video_path(c2ws_all[pair_idx], N_views=40)# you can enlarge the rads_scale if you want to render larger baseline
#             # c2ws_render = create_spheric_poses(2, n_poses=10)
#             # c2ws_render = render_path_imgs(c2ws_all[pair_idx])
#         else:            
#             # neighboring views with position distance
#             imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(device=device)
#             volume_feature, _, _ = MVSNet(imgs_source, proj_mats, near_far_source, pad=pad, lindisp=args.use_disp)
            
#             pad *= args.imgScale_test
#             w2cs, c2ws = pose_source['w2cs'], pose_source['c2ws']
#             pair_idx = torch.load('configs/pairs.th')[f'{scene}_train']
#             c2ws_render = get_spiral(c2ws_all[pair_idx], near_far_source, rads_scale = 0.1, N_views=60)# you can enlarge the rads_scale if you want to render larger baseline
            
#         c2ws_render = torch.from_numpy(np.stack(c2ws_render)).float().to(device)

            
#         imgs_source = unpreprocess(imgs_source)

#         try:
#             tqdm._instances.clear() 
#         except Exception:     
#             pass
        
#         frames = []
#         img_directions = dataset.directions.to(device)
#         for i, c2w in enumerate(tqdm(c2ws_render)):
#             torch.cuda.empty_cache()
            
#             rays_o, rays_d = get_rays(img_directions, c2w)  # both (h*w, 3)
#             rays = torch.cat([rays_o, rays_d,
#                      near_far_source[0] * torch.ones_like(rays_o[:, :1]),
#                      near_far_source[1] * torch.ones_like(rays_o[:, :1])],
#                     1).to(device)  # (H*W, 3)
            
            
#             N_rays_all = rays.shape[0]
#             rgb_rays, depth_rays_preds = [],[]
#             for chunk_idx in range(N_rays_all//args.chunk + int(N_rays_all%args.chunk>0)):

#                 xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
#                                                     N_samples=args.N_samples, lindisp=args.use_disp)

#                 # Converting world coordinate to ndc coordinate
#                 H, W = imgs_source.shape[-2:]
#                 inv_scale = torch.tensor([W - 1, H - 1]).to(device)
#                 w2c_ref, intrinsic_ref = pose_source['w2cs'][0], pose_source['intrinsics'][0].clone()
#                 xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
#                                              near=near_far_source[0], far=near_far_source[1], pad=pad, lindisp=args.use_disp)


#                 # rendering
#                 rgb, disp, acc, depth_pred, alpha, extras = rendering(args, pose_source, xyz_coarse_sampled,
#                                                                        xyz_NDC, z_vals, rays_o, rays_d,
#                                                                        volume_feature,imgs_source, **render_kwargs_train)
    
#                 rgb, depth_pred = torch.clamp(rgb.cpu(),0,1.0).numpy(), depth_pred.cpu().numpy()
#                 rgb_rays.append(rgb)
#                 depth_rays_preds.append(depth_pred)

            
#             depth_rays_preds = np.concatenate(depth_rays_preds).reshape(H, W)
#             depth_rays_preds, _ = visualize_depth_numpy(depth_rays_preds, near_far_source)
            
#             rgb_rays = np.concatenate(rgb_rays).reshape(H, W, 3)
#             H_crop, W_crop = np.array(rgb_rays.shape[:2])//20
#             rgb_rays = rgb_rays[H_crop:-H_crop,W_crop:-W_crop]
#             depth_rays_preds = depth_rays_preds[H_crop:-H_crop,W_crop:-W_crop]
#             # print(depth_rays_preds)
#             # img_vis = np.concatenate((rgb_rays*255,depth_rays_preds),axis=1)
#             img_vis = rgb_rays*255
            
#             imageio.imwrite(f'{save_dir}/{str(i).zfill(4)}.png', img_vis.astype('uint8'))
#             frames.append(img_vis.astype('uint8'))


#     imageio.mimwrite(f'{save_dir}/ft_{scene}_spiral_test_v4.mp4', np.stack(frames), fps=10, quality=10)




# for i_scene, scene in enumerate(['ship','drums','ficus','materials',    'hotdog','lego','mic','chair']):#

#     cmd = f'--datadir /dataset/mvsnerf/nerf_synthetic/{scene}\
#      --dataset_name blender --white_bkgd --imgScale_test {1.0} '

#     is_finetued = False # set True if rendering with finetuning
#     if is_finetued:
#         cmd += f'--ckpt ./runs_fine_tuning/{scene}_1h/ckpts//latest.tar'
#         pad = 0 #the padding value should be same as your finetuning ckpt
#     else:
#         cmd += '--ckpt ./ckpts//mvsnerf-v0.tar'
#         pad = 24 #the padding value should be same as your finetuning ckpt
        
#     args = config_parser(cmd.split())
#     args.use_viewdirs = True

#     args.N_samples = 128
#     args.feat_dim = 8+3*4
# #     args.use_color_volume = False if not is_finetued else args.use_color_volume

#     # create models
#     if i_scene==0 or is_finetued:
#         render_kwargs_train, render_kwargs_test, start, grad_vars = create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True)
#         filter_keys(render_kwargs_train)

#         MVSNet = render_kwargs_train['network_mvs']
#         render_kwargs_train.pop('network_mvs')


#     datadir = args.datadir
#     datatype = 'val'
#     args.chunk = 5120
#     frames = 60


#     dataset = dataset_dict[args.dataset_name](args, split=datatype)
#     val_idx = dataset.img_idx
    
#     save_as_image = False
#     save_dir = f'results/nerf'
#     os.makedirs(save_dir, exist_ok=True)
#     MVSNet.train()
#     MVSNet = MVSNet.cuda()
    
#     with torch.no_grad():

#         if is_finetued:   
#             # large baselien
#             imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(device=device)
#             volume_feature = torch.load(args.ckpt)['volume']['feat_volume']
#             volume_feature = RefVolume(volume_feature.detach()).cuda()
#             c2ws_render = nerf_video_path(pose_source['c2ws'].cpu(), N_views=frames)
#         else:            
#             # neighboring views with angle distance
#             c2ws_all = dataset.load_poses_all()
#             random_selete = torch.randint(0,len(c2ws_all),(1,))     #!!!!!!!!!! you may change this line if rendering a specify view 
#             dis = np.sum(c2ws_all[:,:3,2] * c2ws_all[[random_selete],:3,2], axis=-1)
#             pair_idx = np.argsort(dis)[::-1][torch.randperm(5)[:3]]
#             imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(device=device)#pair_idx=pair_idx, 
#             volume_feature, _, _ = MVSNet(imgs_source, proj_mats, near_far_source, pad=pad)
            
#             #####
#             c2ws_render = gen_render_path(c2ws_all[pair_idx], N_views=frames)
#             c2ws_render = torch.from_numpy(np.stack(c2ws_render)).float().to(device)

            
#         imgs_source = unpreprocess(imgs_source)
        

#         try:
#             tqdm._instances.clear() 
#         except Exception:     
#             pass
        
#         frames = []
#         img_directions = dataset.directions.to(device)
#         for i, c2w in enumerate(tqdm(c2ws_render)):
#             torch.cuda.empty_cache()
            
#             rays_o, rays_d = get_rays(img_directions, c2w)  # both (h*w, 3)
#             rays = torch.cat([rays_o, rays_d,
#                      near_far_source[0] * torch.ones_like(rays_o[:, :1]),
#                      near_far_source[1] * torch.ones_like(rays_o[:, :1])],
#                     1).to(device)  # (H*W, 3)
            
        
#             N_rays_all = rays.shape[0]
#             rgb_rays, depth_rays_preds = [],[]
#             for chunk_idx in range(N_rays_all//args.chunk + int(N_rays_all%args.chunk>0)):

#                 xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
#                                                     N_samples=args.N_samples)

#                 # Converting world coordinate to ndc coordinate
#                 H, W = imgs_source.shape[-2:]
#                 inv_scale = torch.tensor([W - 1, H - 1]).to(device)
#                 w2c_ref, intrinsic_ref = pose_source['w2cs'][0], pose_source['intrinsics'][0].clone()
#                 xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
#                                              near=near_far_source[0], far=near_far_source[1], pad=pad*args.imgScale_test)


#                 # rendering
#                 rgb, disp, acc, depth_pred, alpha, extras = rendering(args, pose_source, xyz_coarse_sampled,
#                                                                        xyz_NDC, z_vals, rays_o, rays_d,
#                                                                        volume_feature,imgs_source, **render_kwargs_train)
    
                
#                 rgb, depth_pred = torch.clamp(rgb.cpu(),0,1.0).numpy(), depth_pred.cpu().numpy()
#                 rgb_rays.append(rgb)
#                 depth_rays_preds.append(depth_pred)

            
#             depth_rays_preds = np.concatenate(depth_rays_preds).reshape(H, W)
#             depth_rays_preds, _ = visualize_depth_numpy(depth_rays_preds, near_far_source)
            
#             rgb_rays = np.concatenate(rgb_rays).reshape(H, W, 3)
#             H_crop, W_crop = np.array(rgb_rays.shape[:2])//20
# #             rgb_rays = rgb_rays[H_crop:-H_crop,W_crop:-W_crop]
# #             depth_rays_preds = depth_rays_preds[H_crop:-H_crop,W_crop:-W_crop]
#             img_vis = np.concatenate((rgb_rays*255,depth_rays_preds),axis=1)

#             frames.append(img_vis.astype('uint8'))
# #             break
#     imageio.mimwrite(f'{save_dir}/ft_{scene}_spiral.mp4', np.stack(frames), fps=10, quality=10)
# # plt.imshow(rgb_rays)



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


# for i_scene, scene in enumerate([1]):# any scene index, like 1,2,3...,,8,21,103,114

#     cmd = f'--datadir /dataset/mvsnerf/mvs_training/dtu/scan{scene} \
#      --dataset_name dtu_ft --imgScale_test {1.0} ' #--use_color_volume
    
#     is_finetued = False # set False if rendering without finetuning
#     if is_finetued:
#         cmd += f'--ckpt ./runs_fine_tuning/scan{scene}-ft/ckpts//latest.tar'
#     else:
#         cmd += '--ckpt ./169999.tar'

#     args = config_parser(cmd.split())
#     args.use_viewdirs = True

#     args.N_samples = 128
#     args.feat_dim =  8+3*4
#     args.use_color_volume = False if not is_finetued else args.use_color_volume


#     # create models
#     if i_scene==0 or is_finetued:
#         render_kwargs_train, render_kwargs_test, start, grad_vars = create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True)
#         filter_keys(render_kwargs_train)

#         MVSNet = render_kwargs_train['network_mvs']
#         render_kwargs_train.pop('network_mvs')


#     datadir = args.datadir
#     datatype = 'val'
#     pad = 24 #the padding value should be same as your finetuning ckpt
#     args.chunk = 5120
#     frames = 60


#     dataset = dataset_dict[args.dataset_name](args, split=datatype)
#     val_idx = dataset.img_idx
    
#     save_as_image = False
#     save_dir = f'results/dtu'
#     os.makedirs(save_dir, exist_ok=True)
#     MVSNet.train()
#     MVSNet = MVSNet.cuda()
    
#     with torch.no_grad():
        
#         if is_finetued:   
#             # large baselien
#             imgs_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(device=device)
#             volume_feature = torch.load(args.ckpt)['volume']['feat_volume']
#             volume_feature = RefVolume(volume_feature.detach()).cuda()
#         else:            
#             # neighboring views with angle distance
#             c2ws_all = dataset.load_poses_all()
#             random_selete = torch.randint(0,len(c2ws_all),(1,)) #!!!!!!!!!! you may change this line if rendering a specify view 
#             dis = np.sum(c2ws_all[:,:3,2] * c2ws_all[[random_selete],:3,2], axis=-1)
#             pair_idx = np.argsort(dis)[::-1][:3]#[25, 21, 33]#[14,15,24]#
#             imges_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(pair_idx=pair_idx, device=device)
#             volume_feature, img_feat, _ = MVSNet(imges_source, proj_mats, near_far_source, pad=pad)
            
#         imgs_source = unpreprocess(imges_source)

#         c2ws_render = gen_render_path(pose_source['c2ws'].cpu().numpy(), N_views=frames)
#         c2ws_render = torch.from_numpy(np.stack(c2ws_render)).float().to(device)
        
        
#         try:
#             tqdm._instances.clear() 
#         except Exception:     
#             pass
        
#         frames = []
#         img_directions = dataset.directions.to(device)
#         for i, c2w in enumerate(tqdm(c2ws_render)):
#             torch.cuda.empty_cache()
            
#             rays_o, rays_d = get_rays(img_directions, c2w)  # both (h*w, 3)
#             rays = torch.cat([rays_o, rays_d,
#                      near_far_source[0] * torch.ones_like(rays_o[:, :1]),
#                      near_far_source[1] * torch.ones_like(rays_o[:, :1])],
#                     1).to(device)  # (H*W, 3)
            
        
#             N_rays_all = rays.shape[0]
#             rgb_rays, depth_rays_preds = [],[]
#             for chunk_idx in range(N_rays_all//args.chunk + int(N_rays_all%args.chunk>0)):

#                 xyz_coarse_sampled, rays_o, rays_d, z_vals = ray_marcher(rays[chunk_idx*args.chunk:(chunk_idx+1)*args.chunk],
#                                                     N_samples=args.N_samples)

#                 # Converting world coordinate to ndc coordinate
#                 H, W = imgs_source.shape[-2:]
#                 H, W = int(H), int(W)
#                 inv_scale = torch.tensor([W - 1, H - 1]).to(device)
#                 w2c_ref, intrinsic_ref = pose_source['w2cs'][0], pose_source['intrinsics'][0].clone()
#                 xyz_NDC = get_ndc_coordinate(w2c_ref, intrinsic_ref, xyz_coarse_sampled, inv_scale,
#                                              near=near_far_source[0], far=near_far_source[1], pad=pad*args.imgScale_test)
#                 ndc_3d = get_ndc_coordinate(None, intrinsic_ref, xyz_coarse_sampled, inv_scale,
#                                                 near=near_far_source[0], far=near_far_source[1], pad=pad*args.imgScale_test)
#                 feat_sampled = compute_projections(imges_source, xyz_coarse_sampled, pose_source, img_feat, H, W)
                    

#                 # rendering
#                 # rgb, disp, acc, depth_pred, alpha, extras = rendering(args, pose_source, xyz_coarse_sampled,
#                 #                                                        xyz_NDC, z_vals, rays_o, rays_d,
#                 #                                                        volume_feature,imgs_source, **render_kwargs_train)
#                 rgb, disp, acc, depth_map_fn, depth_pred, alpha, extras = rendering(args,feat_sampled,ndc_3d, pose_source, xyz_coarse_sampled,
#                                                                         xyz_NDC, z_vals, rays_o, rays_d,
#                                                                         volume_feature,imgs_source, **render_kwargs_train)
                
#                 rgb, depth_pred = torch.clamp(rgb.cpu(),0,1.0).numpy(), depth_pred.cpu().numpy()
#                 rgb_rays.append(rgb)
#                 depth_rays_preds.append(depth_pred)

            
#             depth_rays_preds = np.concatenate(depth_rays_preds).reshape(H, W)
#             depth_rays_preds, _ = visualize_depth_numpy(depth_rays_preds, near_far_source)
            
#             rgb_rays = np.concatenate(rgb_rays).reshape(H, W, 3)
# #             H_crop, W_crop = np.array(rgb_rays.shape[:2])//20
# #             rgb_rays = rgb_rays[H_crop:-H_crop,W_crop:-W_crop]
# #             depth_rays_preds = depth_rays_preds[H_crop:-H_crop,W_crop:-W_crop]
#             img_vis = np.concatenate((rgb_rays*255,depth_rays_preds),axis=1)
#             frames.append(img_vis.astype('uint8'))
                
#     imageio.mimwrite(f'{save_dir}/ft_scan{scene}_spiral2.mp4', np.stack(frames), fps=20, quality=10)
# # plt.imshow(rgb_rays)




cmd = f'--datadir /dataset/mvsnerf/spaces_dataset \
    --dataset_name spaces --imgScale_test {1.0} ' #--use_color_volume

is_finetued = False # set False if rendering without finetuning
# if is_finetued:
#     cmd += f'--ckpt ./runs_fine_tuning/scan-ft/ckpts//latest.tar'
# else:
cmd += '--ckpt ./169999.tar'

args = config_parser(cmd.split())
args.use_viewdirs = True

args.N_samples = 128
args.feat_dim =  8+3*4
args.use_color_volume = False if not is_finetued else args.use_color_volume


# create models

render_kwargs_train, render_kwargs_test, start, grad_vars = create_nerf_mvs(args, use_mvs=True, dir_embedder=False, pts_embedder=True)
filter_keys(render_kwargs_train)

MVSNet = render_kwargs_train['network_mvs']
render_kwargs_train.pop('network_mvs')


datadir = args.datadir
datatype = 'val'
pad = 24 #the padding value should be same as your finetuning ckpt
args.chunk = 5120
frames = 60

# from torch.utils.data import DataLoader
dataset = dataset_dict[args.dataset_name](dataset_path=args.datadir, is_val=True, layout='small_4_1',  # small_4_1 # medium_4_1 # difficult_4_1
                                                n_planes=args.N_samples, im_w=800, im_h=480)
# val_idx = dataset.img_idx
val_dataset = DataLoader(dataset,
            shuffle=False,
            num_workers=0,
            batch_size=1,
            pin_memory=True)
save_as_image = False
save_dir = f'results/spaces'
os.makedirs(save_dir, exist_ok=True)
MVSNet.train()
MVSNet = MVSNet.cuda()

def decode_batch_spaces(batch, idx=list(torch.arange(4))):

    data_mvs = sub_selete_data(batch, device, idx, filtKey=[])
    pose_ref = {'w2cs': data_mvs['w2cs'].squeeze(), 'intrinsics': data_mvs['intrinsics'].squeeze(),
                'c2ws': data_mvs['c2ws'].squeeze(), 'near_fars': data_mvs['near_fars'].squeeze()}

    return data_mvs, pose_ref

with torch.no_grad():
    for i, batch in enumerate(val_dataset):
        if i == 0:
            data_mvs, pose_ref = decode_batch_spaces(batch)
            break
    # data_mvs, pose_ref = val_dataset[0]
    imges, proj_mats = data_mvs['images'], data_mvs['proj_mats'] 
    near_fars = pose_ref['near_fars']
    c2ws_all_t = data_mvs['c2ws_all'][0]
    # print(c2ws_all.shape)
    # print(near_fars[0])
    # exit()
    # random_selete = torch.randint(0,len(c2ws_all_t),(1,)) #!!!!!!!!!! you may change this line if rendering a specify view 
    # # print(random_selete)
    c2ws_all = c2ws_all_t.cpu().numpy()
    # temp = c2ws_all[:,:3,2] * c2ws_all[random_selete,:3,2]
    # print(temp.shape)
    # dis = np.sum(c2ws_all[:,:3,2] * c2ws_all[random_selete,:3,2], axis=-1)
    # pair_idx = np.argsort(dis)[::-1][:3]#[25, 21, 33]#[14,15,24]#
    # imges_source, proj_mats, near_far_source, pose_source = dataset.read_source_views(pair_idx=pair_idx, device=device)
    imges_source = imges[:, :3]
    proj_mats = proj_mats[:, :3]
    volume_feature, img_feat, _ = MVSNet(imges_source, proj_mats, near_fars[0], pad=pad)
        
    imgs_source = unpreprocess(imges_source)

    c2ws_render = gen_render_path(c2ws_all, N_views=frames)
    c2ws_render = torch.from_numpy(np.stack(c2ws_render)).float().to(device)
    world_to_ref = pose_ref['w2cs'][0]
    tgt_to_world, intrinsic = pose_ref['c2ws'][-1], pose_ref['intrinsics'][-1]
    try:
        tqdm._instances.clear() 
    except Exception:     
        pass
    H, W = imges.shape[-2:]
    frames = []
    # img_directions = dataset.directions.to(device)
    for i, c2w in enumerate(tqdm(c2ws_render)):
        torch.cuda.empty_cache()
        rgb_rays, depth_rays_preds = [],[]
        for chunk_idx in range(H * W//args.chunk + int(H * W%args.chunk>0)):
            rays_pts, rays_dir, rays_NDC, depth_candidates, rays_o, ndc_parameters, ndc_3d = \
                    build_rays_test(H, W, c2w, world_to_ref, intrinsic, near_fars, near_fars[-1], args.N_samples, pad=args.pad, chunk=args.chunk, idx=chunk_idx)
            feat_sampled = compute_projections(imges, rays_pts, pose_ref, img_feat, H, W)
                
            rgb, disp, acc, depth_map_fn, depth_pred, density_ray, ret = rendering(args, feat_sampled, ndc_3d, pose_ref, rays_pts, rays_NDC, depth_candidates, rays_o, rays_dir,
                                                       volume_feature, imgs_source, img_feat=None,  **render_kwargs_train)

            rgb, depth_pred = torch.clamp(rgb.cpu(), 0, 1.0).numpy(), depth_pred.cpu().numpy()
            rgb_rays.append(rgb)
            depth_rays_preds.append(depth_pred)
        
        depth_rays_preds = np.concatenate(depth_rays_preds).reshape(H, W)
        depth_rays_preds, _ = visualize_depth_numpy(depth_rays_preds, near_fars[0].cpu().numpy())
        
        rgb_rays = np.concatenate(rgb_rays).reshape(H, W, 3)
#             H_crop, W_crop = np.array(rgb_rays.shape[:2])//20
#             rgb_rays = rgb_rays[H_crop:-H_crop,W_crop:-W_crop]
#             depth_rays_preds = depth_rays_preds[H_crop:-H_crop,W_crop:-W_crop]
        img_vis = np.concatenate((rgb_rays*255,depth_rays_preds),axis=1)
        imageio.imwrite(f'{save_dir}/{str(i).zfill(4)}.png', img_vis.astype('uint8'))
        frames.append(img_vis.astype('uint8'))
        
imageio.mimwrite(f'{save_dir}/spaces.mp4', np.stack(frames), fps=20, quality=10)

