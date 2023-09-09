import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from .ray_utils import *
import imageio
from .pose_utils import (
    interpolate_poses,
    correct_poses_bounds,
    create_spiral_poses,
)
import copy

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

import numpy as np
import os
import imageio


class MVSDatasetShiny(Dataset):
    def __init__(self, args, split='train', spheric_poses=True, load_ref=False):
        self.args = args
        self.root_dir = args.datadir
        self.split = split
        self.img_wh = (640, 512)  # 960 640
        assert self.img_wh[0] % 32 == 0 or self.img_wh[1] % 32 == 0, \
            'image width must be divisible by 32, you may need to modify the imgScale'
        self.spheric_poses = spheric_poses
        self.define_transforms()

        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # self.blender2opencv = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.length_data = []
        self.scens_item_length = []
        if not load_ref:
            self.read_meta()
        self.white_back = False

    def read_meta(self):
        list_scen = os.listdir(self.root_dir)
        list_scen.sort()
        # print(list_scen)
        if self.split == 'train':
            scens = list_scen[:-1]
        else:
            scens = list_scen[:]
        self.select = []
        near_far_all, proj_mat_ls_all = [], []
        intrinsics_all, c2ws_all, w2cs_all = [], [], []
        self.scens_item_length.append([0])
        for temp in range(len(scens)):
            
            scen = scens[temp]
            # print(scen)
            scene_path = os.path.join(self.root_dir, scen)
            
            poses_bounds = np.load(os.path.join(scene_path, 'poses_bounds.npy'))
            poses = poses_bounds[:, :12].reshape(-1, 3, 4)
            bounds = poses_bounds[:, -2:]
            # print(poses.shape, intrinsic)
            poses, poses_avg, bounds = correct_poses_bounds(poses, bounds, use_train_pose=True)
            # print(poses[0])
            # poses = recenter_poses(poses)
            intrinsic_arr = np.load(os.path.join(scene_path, 'hwf_cxcy.npy'))
            H, W, focal_x, focal_y, cx, cy = intrinsic_arr[:, 0]
            # print(H, W, focal_x, focal_y, cx, cy)
            with open(os.path.join(self.root_dir, scen, 'planes.txt'), "r") as fi:
                data = [float(x) for x in fi.readline().split(" ")]
                dmin, dmax = data[0], data[1]
                bds = [0.95*dmin, 1.05*dmax]
            
            image_dir = os.path.join(scene_path, "images")
            rgb_files = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
            lenth_all = len(poses) + self.scens_item_length[temp][0]
            # select_idx = np.random.randint(0, len(poses), 4)
            select_idx = np.array([i for i in range(0, len(poses), 10)])
            self.select.extend(select_idx + self.scens_item_length[temp][0])
            self.scens_item_length.append([lenth_all])
            # print(select_idx)
            # print(self.select)
            scale_x, scale_y = self.img_wh[0] / W, self.img_wh[1] / H
            fx = focal_x * scale_x
            fy = focal_y * scale_y
            cx = cx * scale_x
            cy = cy * scale_y
            
            # Step 2: correct poses
            near_far = [bds[0], bds[1]]

            for i in range(len(poses)):
                
                intrinsic = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                self.length_data.append([rgb_files[i], temp, i])  # scens, img
                w2c = np.eye(4)
                w2c[:3, :4] = poses[i]
                c2w = np.linalg.inv(w2c)
                c2ws_all.append(c2w)
                w2cs_all.append(w2c)
                # build proj mat from source views to ref view
                proj_mat_l = np.eye(4)
                
                intrinsics_all.append(intrinsic.copy())
                intrinsic[:2] = intrinsic[:2] / 4
                proj_mat_l[:3, :4] = intrinsic @ w2c[:3, :4]

                proj_mat_ls_all.append(proj_mat_l)
                
                near_far_all.append([[near_far[0], near_far[1]]])
        self.near_far, self.intrinsics = near_far_all, intrinsics_all
        self.world2cams, self.cam2worlds = w2cs_all, c2ws_all
        self.proj_mat_ls = proj_mat_ls_all

    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                                    ])

    def __len__(self):
        return len(self.select)

    def __getitem__(self, idx):
        sample = {}
        idx = self.select[idx]
        # print(idx)
        img_name, scen, img_idx = self.length_data[idx]  # scen:0 img_idx:0
        length_low, length_high = self.scens_item_length[scen][0], self.scens_item_length[scen + 1][0]
        idx_low = max(length_low, idx - 5)
        idx_high = min(length_high - 1, idx + 5)
        # print(idx, length_low, length_high, idx_low, idx_high)
        temp = 0
        ref_idx = []
        while temp < 3:
            ids = np.random.randint(idx_low, idx_high)
            if ids != idx and ids not in ref_idx:
                ref_idx.append(ids)
                temp += 1
        affine_mat, affine_mat_inv = [], []
        
        imgs, depths_h = [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
        ref_idx.append(idx)
        # print("range")
        for i in range(4):
            # print(i)
            index = ref_idx[i]
            img_name = self.length_data[index][0]
            img = Image.open(img_name).convert('RGB')
            img = img.resize(self.img_wh, Image.BILINEAR)
            img = self.transform(img)
            imgs += [img]
            proj_mat_ls = self.proj_mat_ls[index]
            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            
            near_far = self.near_far[index][0]
            intrinsics.append(self.intrinsics[index])
            w2cs.append(self.world2cams[index])
            c2ws.append(self.cam2worlds[index])
            depths_h.append(np.zeros((1, 1)))
            near_fars.append(near_far)
            if i == 0:
                ref_proj_inv = np.linalg.inv(proj_mat_ls)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]
        imgs = torch.stack(imgs).float()
        depths_h = np.stack(depths_h)
        proj_mats = np.stack(proj_mats)[:, :3]  # 修改
        
        affine_mat, affine_mat_inv = np.stack(affine_mat), np.stack(affine_mat_inv)

        w2cs = np.stack(w2cs)
        c2ws = np.stack(c2ws)
        near_fars = np.stack(near_fars)
        intrinsics = np.stack(intrinsics)

        sample['images'] = imgs  # (V, H, W, 3)
        sample['depths_h'] = depths_h.astype(np.float32)  # (V, H, W)
        sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample['near_fars'] = near_fars.astype(np.float32)
        sample['proj_mats'] = proj_mats.astype(np.float32)

        sample['intrinsics'] = intrinsics.astype(np.float32)  # (V, 3, 3)
        sample['affine_mat'] = affine_mat
        sample['affine_mat_inv'] = affine_mat_inv
        return sample

# import torch
# from torch.utils.data import Dataset
# import glob
# import numpy as np
# import os
# from PIL import Image
# from torchvision import transforms as T
# from .ray_utils import *
# from .shiny_data_utils import load_llff_data
# from .pose_utils import *


# class MVSDatasetShiny(Dataset):
#     def __init__(self, args, split='train', spheric_poses=True, load_ref=False):
#         self.args = args
#         self.root_dir = args.datadir
#         self.split = split
#         self.img_wh = (960, 640)  # 960 640
#         assert self.img_wh[0] % 32 == 0 or self.img_wh[1] % 32 == 0, \
#             'image width must be divisible by 32, you may need to modify the imgScale'
#         self.spheric_poses = spheric_poses
#         self.define_transforms()

#         # self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#         # self.blender2opencv = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
#         self.length_data = []
#         self.scens_item_length = []
#         if not load_ref:
#             self.read_meta()
#         self.white_back = False

#     def read_meta(self):
#         list_scen = os.listdir(self.root_dir)
#         if self.split == 'train':
#             scens = list_scen[0:-1]
#         else:
#             scens = list_scen[-1:]
#         proj_mats_all4, near_far_all, proj_mat_ls_all4 = [], [], []
#         proj_mats_all8, proj_mat_ls_all8 = [], []
#         proj_mats_all16, proj_mat_ls_all16 = [], []
#         intrinsics_all, c2ws_all, w2cs_all = [], [], []

#         self.scens_item_length.append([0])
#         for temp in range(len(scens)):
#             scen = scens[temp]
#             scene_path = os.path.join(self.root_dir, scen)
#             poses = np.load(os.path.join(self.root_dir, scen, 'poses_bounds.npy'))
#             poses = poses[:, :-2].reshape(-1, 3, 4)
#             H, W, fx, fy, cx, cy = np.load(os.path.join(self.root_dir, scen, 'hwf_cxcy.npy'))
#             image_dir = os.path.join(scene_path, "images")
#             with open(os.path.join(self.root_dir, scen, 'planes.txt'), "r") as fi:
#                 data = [float(x) for x in fi.readline().split(" ")]
#                 dmin, dmax = data[0], data[1]
#                 bds = [0.95*dmin, 1.05*dmax]
#             rgb_files = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
#             lenth_all = len(poses) + self.scens_item_length[temp][0]
#             self.scens_item_length.append([lenth_all])
#             scale_x, scale_y = self.img_wh[0] / W, self.img_wh[1] / H
#             H = H * scale_y
#             W = W * scale_x
#             fx = fx * scale_x
#             fy = fy * scale_y
#             cx = cx * scale_x
#             cy = cy * scale_y
#             intrinsic = np.array(
#                 [[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
#             # Step 2: correct poses
#             near_far = [bds[0], bds[1]]

#             for i in range(len(poses)):
#                 self.length_data.append([rgb_files[i], temp, i])  # scens, img
#                 c2w = np.eye(4)
#                 c2w[:3] = poses[i]
#                 w2c = np.linalg.inv(c2w)
#                 c2ws_all.append(c2w)
#                 w2cs_all.append(w2c)
#                 # build proj mat from source views to ref view
#                 proj_mat_l4 = np.eye(4)
#                 proj_mat_l8 = np.eye(4)
#                 proj_mat_l16 = np.eye(4)
                
#                 intrinsics_all.append(intrinsic.copy())
#                 intrinsic[:2] = intrinsic[:2] / 4  # 4 times downscale in the feature space
#                 proj_mat_l4[:3, :4] = intrinsic @ w2c[:3, :4]
#                 proj_mat_ls_all4.append(proj_mat_l4)
                
#                 intrinsic[:2] = intrinsic[:2] / 2  # 4 times downscale in the feature space
#                 proj_mat_l8[:3, :4] = intrinsic @ w2c[:3, :4]
#                 proj_mat_ls_all8.append(proj_mat_l8)
                
#                 intrinsic[:2] = intrinsic[:2] / 2  # 4 times downscale in the feature space
#                 proj_mat_l16[:3, :4] = intrinsic @ w2c[:3, :4]
#                 proj_mat_ls_all16.append(proj_mat_l16)
                
#                 near_far_all.append([[near_far[0], near_far[1]]])
#         self.near_far, self.intrinsics = near_far_all, intrinsics_all
#         self.world2cams, self.cam2worlds = w2cs_all, c2ws_all
#         self.proj_mat_ls4 = proj_mat_ls_all4
#         self.proj_mat_ls8 = proj_mat_ls_all8
#         self.proj_mat_ls16 = proj_mat_ls_all16

#     def define_transforms(self):
#         self.transform = T.Compose([T.ToTensor(),
#                                     T.Normalize(mean=[0.485, 0.456, 0.406],
#                                                 std=[0.229, 0.224, 0.225]),
#                                     ])

#     def __len__(self):
#         return len(self.length_data)

#     def __getitem__(self, idx):
#         sample = {}
#         img_name, scen, img_idx = self.length_data[idx]  # scen:0 img_idx:0
#         length_low, length_high = self.scens_item_length[scen][0], self.scens_item_length[scen + 1][0]
#         temp = 0
#         ref_idx = []
#         while temp < 3:
#             ids = np.random.randint(length_low, length_high - 1)
#             if ids != idx:
#                 ref_idx.append(ids)
#                 temp += 1
#                 # print(temp)
#         affine_mat4, affine_mat_inv4 = [], []
#         affine_mat8, affine_mat_inv8 = [], []
#         affine_mat16, affine_mat_inv16 = [], []
#         proj_mats8 = []
#         proj_mats16 = []
#         imgs, depths_h = [], []
#         proj_mats4, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
#         ref_idx.append(idx)
#         # print("range")
#         for i in range(4):
#             # print(i)
#             index = ref_idx[i]
#             img_name = self.length_data[index][0]
#             img = Image.open(img_name).convert('RGB')
#             img = img.resize(self.img_wh, Image.BILINEAR)
#             img = self.transform(img)
#             imgs += [img]
#             proj_mat_ls4 = self.proj_mat_ls4[index]
#             affine_mat4.append(proj_mat_ls4)
#             affine_mat_inv4.append(np.linalg.inv(proj_mat_ls4))
            
#             proj_mat_ls8 = self.proj_mat_ls8[index]
#             affine_mat8.append(proj_mat_ls8)
#             affine_mat_inv8.append(np.linalg.inv(proj_mat_ls8))
            
#             proj_mat_ls16 = self.proj_mat_ls16[index]
#             affine_mat16.append(proj_mat_ls16)
#             affine_mat_inv16.append(np.linalg.inv(proj_mat_ls16))
            
#             near_far = self.near_far[index][0]
#             intrinsics.append(self.intrinsics[index])
#             w2cs.append(self.world2cams[index])
#             c2ws.append(self.cam2worlds[index])
#             depths_h.append(np.zeros((1, 1)))
#             near_fars.append(near_far)
#             if i == 0:
#                 ref_proj_inv4 = np.linalg.inv(proj_mat_ls4)
#                 proj_mats4 += [np.eye(4)]
#                 ref_proj_inv8 = np.linalg.inv(proj_mat_ls8)
#                 proj_mats8 += [np.eye(4)]
#                 ref_proj_inv16 = np.linalg.inv(proj_mat_ls16)
#                 proj_mats16 += [np.eye(4)]
#             else:
#                 proj_mats4 += [proj_mat_ls4 @ ref_proj_inv4]
#                 proj_mats8 += [proj_mat_ls8 @ ref_proj_inv8]
#                 proj_mats16 += [proj_mat_ls16 @ ref_proj_inv16]

#         imgs = torch.stack(imgs).float()
#         depths_h = np.stack(depths_h)
#         proj_mats4 = np.stack(proj_mats4)[:, :3]  # 修改
#         proj_mats8 = np.stack(proj_mats8)[:, :3]
#         proj_mats16 = np.stack(proj_mats16)[:, :3]
#         affine_mat4, affine_mat_inv4 = np.stack(affine_mat4), np.stack(affine_mat_inv4)
#         affine_mat8, affine_mat_inv8 = np.stack(affine_mat8), np.stack(affine_mat_inv8)
#         affine_mat16, affine_mat_inv16 = np.stack(affine_mat16), np.stack(affine_mat_inv16)
#         w2cs = np.stack(w2cs)
#         c2ws = np.stack(c2ws)
#         near_fars = np.stack(near_fars)
#         intrinsics = np.stack(intrinsics)

#         sample['images'] = imgs  # (V, H, W, 3)
#         sample['depths_h'] = depths_h.astype(np.float32)  # (V, H, W)
#         sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
#         sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
#         sample['near_fars'] = near_fars.astype(np.float32)
#         sample['proj_mats4'] = proj_mats4.astype(np.float32)
#         sample['proj_mats8'] = proj_mats8.astype(np.float32)
#         sample['proj_mats16'] = proj_mats16.astype(np.float32)
#         sample['intrinsics'] = intrinsics.astype(np.float32)  # (V, 3, 3)
#         sample['affine_mat4'] = affine_mat4
#         sample['affine_mat_inv4'] = affine_mat_inv4
#         sample['affine_mat8'] = affine_mat8
#         sample['affine_mat_inv8'] = affine_mat_inv8
#         sample['affine_mat16'] = affine_mat16
#         sample['affine_mat_inv16'] = affine_mat_inv16
#         return sample