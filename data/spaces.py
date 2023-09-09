import pathlib
import os
import sys
import random
from PIL import Image
import torch
import torch.utils.data
import numpy as np
from data import utils_dataset
from data.code import utils
from torchvision import transforms as T
import torchvision.transforms.functional as F


class AbstractDsetSpaces(torch.utils.data.Dataset):
    """Abstract Spaces dataset, loads metadata, and contains some useful routines"""

    def __init__(self, dataset_path, is_val=True, n_planes=10, tiny=False, im_w=200, im_h=120):
        """
        dataset_path:数据集文件夹 /dataset/spaces_dataset
        n_planes :最近到最远深度的采样个数
        """

        self.dataset_path = dataset_path
        self.is_val = is_val
        self.n_planes = n_planes
        self.tiny = tiny

        self.n_in = None
        self.n_tgt = None
        self.im_w = im_w
        self.im_h = im_h
        self.image_base_width = 800
        self.image_base_height = 480

        use_2k = False

        # Load metadata
        metadata_base_path = pathlib.Path(dataset_path) / 'data' / (
            '2k' if use_2k else '800')  # /dataset/spaces_dataset/data/800
        dirs = os.listdir(metadata_base_path)  # /dataset/spaces_dataset/data/800/scene_000(...)
        dirs.sort()
        scenes = [utils.ReadScene(metadata_base_path / p) for p in dirs]  # 100 n 16 8
        if not use_2k:
            assert len(scenes) == 100
            if tiny:
                scenes = scenes[:2] if is_val else scenes[10:20]
            else:
                scenes = scenes[26:34] if is_val else scenes[10:]
                # scenes = scenes[92:] if is_val else scenes[0:90]

        self.scenes = scenes
        self.define_transforms()
        # Create the rig total index table, which might be useful
        # i = scene index, j = rig pos in scene
        self.rig_table = []
        n = len(self.scenes)  # 训练：90；验证：10
        for i in range(n):
            for j in range(len(self.scenes[i])):
                self.rig_table.append((i, j))  # scens, imgs(有的多有的少)

    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                                    ])

    def _load_image(self, idx):
        # 外参4*4(w2c==cfw) 乘 世界坐标系得到相机坐标系
        data = self.scenes[idx[0]][idx[1]][idx[2]]
        intrinsic = data.camera.intrinsics
        image_path = data.image_path
        img = Image.open(image_path)
        sy = self.im_h / img.height
        sx = self.im_w / img.width
        intrinsic[0, 0] = intrinsic[0, 0]*sx
        intrinsic[0, 2] = intrinsic[0, 2]*sx
        intrinsic[1, 1] = intrinsic[1, 1]*sy
        intrinsic[1, 2] = intrinsic[1, 2]*sy
        extrinsic = data.camera.c_f_w
        intrin = intrinsic.copy()
        proj_mat_l = np.eye(4)

        intrinsic[:2] = intrinsic[:2] / 4  # 刚加
        proj_mat_l[:3, :4] = intrinsic @ extrinsic[:3, :4]

        img = img.resize((self.im_w, self.im_h), Image.BILINEAR)
        img = self.transform(img)
        return {
            'images': img,
            'intrinsics': intrin,
            'c2ws': np.linalg.inv(extrinsic),
            'w2cs': extrinsic,
            'proj_mats': proj_mat_l,

        }


STANDARD_LAYOUTS = {
    # 'small_4_1': ([2, 1, 11], [6]),
    'small_4_1': ([3,4,12], [8]),
    'medium_4_1': ([3, 1, 12], [7]),
    'difficult_4_1': ([3, 0, 12], [8])
}


#######################################################################################################################
class DsetSpaces(AbstractDsetSpaces):
    def __init__(self, dataset_path, is_val=True, layout='random_4_9', n_planes=10, tiny=False, im_w=200, im_h=120,
                 no_crop=False):
        """
        The "standard" fully deterministic (no randomness) Spaces dataset
        Treat each rig position of each scene as an element, with non-random input and target views
        Different rig positions are not mixed
        For layouts, see Fig.4 of the paper
        :param dataset_path:   Spaces dataset path on disk
        :param is_val:         True=val, False=train
        :param layout:
        """
        super().__init__(dataset_path, is_val, n_planes, tiny, im_w, im_h)
        self.no_crop = no_crop
        self.layout = layout
        if isinstance(layout, str):
            tup = STANDARD_LAYOUTS[layout]
        elif isinstance(layout, tuple):
            tup = layout
        else:
            raise ValueError('DsetSpaces(): layout must be a string or 2-tuple !')
        self.idxs_in, self.idxs_tgt = tup
        self.n_in, self.n_tgt = len(self.idxs_in), len(self.idxs_tgt)
        assert self.n_in > 0 and self.n_tgt > 0

    def __len__(self):
        return len(self.rig_table)

    def __getitem__(self, item):
        sample = {}
        # Add scene and rig to camera indices
        idx_scene, idx_rig = self.rig_table[item]  # 场景，16个相机中对应的图片
        # 训练时
        if not self.is_val:
            # 十六个相机中随机选择输入原图像的个数
            # 随机选择输入的视角
            self.idxs_in = random.choices([i for i in range(0, 16)], k=len(self.idxs_in))
            free_idx = [i for i in range(0, 16) if i not in self.idxs_in]
            tgt_i = random.choices(free_idx, k=len(self.idxs_tgt))
            self.idxs_tgt = tgt_i
            # print(idx_scene, idx_rig, self.idxs_in, tgt_i)
        # self.ref_idx = self.idxs_tgt[0]
        idxs_in_full = [(idx_scene, idx_rig, i) for i in self.idxs_in]  # camera in
        idxs_tgt_full = [(idx_scene, idx_rig, i) for i in self.idxs_tgt]  # target
        affine_mat, affine_mat_inv = [], []

        imgs, depths_h = [], []
        proj_mats, intrinsics, w2cs, c2ws, near_fars = [], [], [], [], []  # record proj mats between views

        for i in range(len(idxs_in_full)):
            res = self._load_image(idxs_in_full[i])
            img = res['images']
            imgs += [img]
            near_fars.append([1, 100])  # 0.5, 60
            intrinsics.append(np.array(res['intrinsics']))
            proj_mat_ls = res['proj_mats']


            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            
            w2cs.append(np.array(res['w2cs']))
            c2ws.append(np.array(res['c2ws']))
            if i == 0:
                ref_proj_inv = np.linalg.inv(proj_mat_ls)
                proj_mats += [np.eye(4)]
                
            else:
                proj_mats += [proj_mat_ls @ ref_proj_inv]
                
            depths_h.append(np.zeros((1, 1)))

        for i in idxs_tgt_full:
            res = self._load_image(i)
            img = res['images']
            imgs += [img]
            near_fars.append([1, 100])

            intrinsics.append(np.array(res['intrinsics']))
            proj_mat_ls = res['proj_mats']
            
            affine_mat.append(proj_mat_ls)
            affine_mat_inv.append(np.linalg.inv(proj_mat_ls))
            
            w2cs.append(np.array(res['w2cs']))
            c2ws.append(np.array(res['c2ws']))
            proj_mats += [proj_mat_ls @ ref_proj_inv]
            
            depths_h.append(np.zeros((1, 1)))
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
