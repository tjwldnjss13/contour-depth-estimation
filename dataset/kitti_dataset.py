import os
import cv2 as cv
import numpy as np
import torch.utils.data as data
import torchvision.transforms as T
import matplotlib.pyplot as plt

from dataset.augment import stereo_random_hsv


class KITTIDataset(data.Dataset):
    def __init__(self, root, size=(384, 1280), mode='train', normalize=False, use_contour=False):
        super().__init__()
        self.root = root
        self.size = size
        self.mode = mode
        self.normalize = normalize
        self.use_contour = use_contour
        self.imgl_pth_list, self.imgr_pth_list = self._get_image_path_list()

    def _get_image_path_list(self):
        l_img_pth_list = []
        r_img_pth_list = []
        img_dir = os.path.join(self.root, 'raw', self.mode)
        sub_dirs = os.listdir(img_dir)
        for sub_dir in sub_dirs:
            dir_l = os.path.join(img_dir, sub_dir, 'image_02', 'data')
            dir_r = os.path.join(img_dir, sub_dir, 'image_03', 'data')
            pths_l = os.listdir(dir_l)
            pths_r = os.listdir(dir_r)
            for p_l in pths_l:
                l_img_pth_list.append(os.path.join(dir_l, p_l))
            for p_r in pths_r:
                r_img_pth_list.append(os.path.join(dir_r, p_r))

        return l_img_pth_list, r_img_pth_list

    def __getitem__(self, idx):
        h, w = self.size
        imgl, imgr = self.imgl_pth_list[idx], self.imgr_pth_list[idx]
        imgl, imgr = cv.imread(imgl), cv.imread(imgr)
        imgl, imgr = cv.cvtColor(imgl, cv.COLOR_BGR2RGB), cv.cvtColor(imgr, cv.COLOR_BGR2RGB)
        imgl, imgr = cv.resize(imgl, (w, h), interpolation=cv.INTER_CUBIC), cv.resize(imgr, (w, h), interpolation=cv.INTER_CUBIC)

        if self.mode == 'train' and np.random.random() < .5:
            imgl, imgr = cv.flip(imgr, 1), cv.flip(imgl, 1)

        if self.use_contour:
            contour = cv.Canny(imgl, 255/3, 255)
            contour = contour > 0

        if self.mode == 'train':
            imgl, imgr = stereo_random_hsv(imgl, imgr)

        imgl, imgr = T.ToTensor()(imgl), T.ToTensor()(imgr)
        if self.normalize:
            imgl = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(imgl)
            imgr = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(imgr)

        dict_ret = {}
        dict_ret['imgl'] = imgl
        dict_ret['imgr'] = imgr
        if self.use_contour:
            contour = T.ToTensor()(contour).float()
            dict_ret['contour'] = contour

        return dict_ret

    def __len__(self):
        return len(self.imgl_pth_list)


def custom_collate_fn(batch):
    # item1 = [item for item in batch]
    # return item1
    return batch


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import Subset
    root = 'C://DeepLearningData/KITTI/'
    dset = KITTIDataset(root, size=(256, 512), do_flip=True, use_contour=True)
    for i in range(len(dset)):
        dset[i]


