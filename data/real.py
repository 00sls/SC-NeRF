
import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
import argparse
from torchvision import transforms as T


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.intrinsics = np.array([[fx, 0, cx, 0],
                                    [0, fy, cy, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def unnormalize_intrinsics(intrinsics, h, w):

    intrinsics[0] *= w
    intrinsics[1] *= h

    return intrinsics

def sub_selete_data(data_batch, device, idx, filtKey=[], filtIndex=['view_ids_all','c2ws_all','scan','bbox','w2ref','ref2w','light_id','ckpt','idx']):
    data_sub_selete = {}
    for item in data_batch.keys():
        data_sub_selete[item] = data_batch[item][:,idx].float() if (item not in filtIndex and torch.is_tensor(item) and item.dim()>2) else data_batch[item].float()
        if not data_sub_selete[item].is_cuda:
            data_sub_selete[item] = data_sub_selete[item].to(device)
    return data_sub_selete

def decode_batch(batch, idx=list(torch.arange(4))):
    data_mvs = sub_selete_data(batch, batch["images"].device, idx, filtKey=[])
    pose_ref = {'w2cs': data_mvs['w2cs'].squeeze(), 'intrinsics': data_mvs['intrinsics'].squeeze(),
                'c2ws': data_mvs['c2ws'].squeeze(), 'near_fars': data_mvs['near_fars'].squeeze()}
    return data_mvs, pose_ref

def parse_pose_file(file):
    f = open(file, 'r')
    cam_params = {}
    for i, line in enumerate(f):
        if i == 0:
            continue
        entry = [float(x) for x in line.split()]
        id = int(entry[0])
        cam_params[id] = Camera(entry)
    return cam_params

class RealEstateDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.folder_path = args.datadir
        self.mode = mode  # train / test / validation
        self.num_source_views = 3
        self.target_h, self.target_w = 512, 640
        assert mode in ['train', 'val']
        self.scene_path_list = glob.glob(os.path.join(self.folder_path, 'imgs', '*'))
        self.scene_path_list.sort()
        print(self.scene_path_list)
        self.define_transforms()
        all_rgb_files = []
        all_timestamps = []
        for i, scene_path in enumerate(self.scene_path_list):
            rgb_files = [os.path.join(scene_path, f) for f in sorted(os.listdir(scene_path))]
            if len(rgb_files) < 10:
                print('omitting {}, too few images'.format(os.path.basename(scene_path)))
                continue
            timestamps = [int(os.path.basename(rgb_file).split('.')[0]) for rgb_file in rgb_files]
            sorted_ids = np.argsort(timestamps)
            select_indx = sorted_ids[::10]
            all_rgb_files.append(np.array(rgb_files)[sorted_ids])
            all_timestamps.append(np.array(timestamps)[sorted_ids])

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files)[index]
        self.all_timestamps = np.array(all_timestamps)[index]

    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                                    ])
    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):
        sample = {}
        rgb_files = self.all_rgb_files[idx]
        timestamps = self.all_timestamps[idx]

        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)
        window_size = 16
        # shift = np.random.randint(low=-1, high=2)
        id_render = int(num_frames//2)  # 渲染视角

        right_bound = min(id_render + window_size, num_frames-1)
        left_bound = max(0, right_bound - 2 * window_size)
        candidate_ids = np.arange(left_bound, right_bound)
        # remove the query frame itself with high probability
        if np.random.choice([0, 1], p=[0.01, 0.99]):
            candidate_ids = candidate_ids[candidate_ids != id_render]
        # print(candidate_ids)
        id_feat = [1, len(candidate_ids)//3, len(candidate_ids)-2]
        # print(id_feat)
        id_feat = candidate_ids[id_feat]
        # id_feat参考视角
        view_ids = [i for i in id_feat] + [id_render]
        print(view_ids)
        camera_file = os.path.dirname(rgb_files[0]).replace('imgs', 'camera') + '.txt'
        cam_params = parse_pose_file(camera_file)
        imgs, depths_h = [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views
        for i, vid in enumerate(view_ids):
            img = Image.open(rgb_files[vid])
            # resize the image to target size
            img = img.resize((self.target_w, self.target_h), Image.LANCZOS)
            rgb = self.transform(img)
            imgs += [rgb]
            cam_param = cam_params[timestamps[vid]]
            intrinsic = unnormalize_intrinsics(cam_param.intrinsics, self.target_h, self.target_w)
            c2w = cam_param.c2w_mat
            w2c = cam_param.w2c_mat
            proj_mat_l = np.eye(4)
            intrinsics.append(intrinsic[:3, :3].copy())
            intrinsic[:2] = intrinsic[:2]/4
            proj_mat_l[:3, :4] = intrinsic[:3, :3] @ w2c[:3, :4]
            
            w2cs.append(w2c)
            c2ws.append(c2w)
            if i == 0:  # reference view
                ref_proj_inv = np.linalg.inv(proj_mat_l)
                proj_mats += [np.eye(4)]
            else:
                proj_mats += [proj_mat_l @ ref_proj_inv]
            near_fars.append([1, 50.])
            depths_h.append(np.zeros((1, 1)))
        depths_h = np.stack(depths_h)
        imgs = torch.stack(imgs).float()
        proj_mats = np.stack(proj_mats)[:, :3]
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(w2cs), np.stack(c2ws), np.stack(near_fars)
        sample['images'] = imgs  # (V, H, W, 3)
        sample['depths_h'] = depths_h.astype(np.float32)  # (V, H, W)
        sample['w2cs'] = w2cs.astype(np.float32)  # (V, 4, 4)
        sample['c2ws'] = c2ws.astype(np.float32)  # (V, 4, 4)
        sample['intrinsics'] = intrinsics.astype(np.float32)  # (V, 3, 3)
        sample['proj_mats'] = proj_mats.astype(np.float32)
        sample['near_fars'] = near_fars.astype(np.float32)
        return sample