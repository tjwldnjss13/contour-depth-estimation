import os
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

from torch.utils.data import DataLoader
from PIL import Image

from dataset.kitti_dataset import *
from models.cdnet2.cdnet2 import CDNet2
from models.cdnet3.cdnet3 import CDNet3
from models.cdnet5.cdnet5 import CDNet5
from models.cdnet6.cdnet6 import CDNet6
from models.cdnet7.cdnet7 import CDNet7
from models.cdnet8.cdnet8 import CDNet8
from utils.disparity import *


def get_dataset_loader():
    root = 'C://DeepLearningData/KITTI/'
    size = (256, 512)
    transform = T.Compose([T.Resize(size), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dset = KITTIDataset(root, size, 'val', transform)
    loader = DataLoader(dset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

    return loader


def get_depth_image(image):
    vmax = np.percentile(image, 95)
    normalizer = colors.Normalize(vmin=image.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma_r')
    depth = (mapper.to_rgba(image)[:, :, :3] * 255).astype(np.uint8)

    return depth


def main(args):
    model = CDNet7().to(args.device)

    ckpt = torch.load(args.weight_pth)
    model.load_state_dict(ckpt['model_state_dict'])

    model.eval()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if len(args.sample_pth_list) == 0:
        loader = get_dataset_loader()
        transform = T.Resize((375, 1242))
        for i, ann in enumerate(loader):
            imgl, imgr = ann[0]['imgl'], ann[0]['imgr']
            imgl, imgr = imgl.unsqueeze(0).to(args.device), imgr.unsqueeze(0).to(args.device)

            disp = model(imgl)[0]
            dr = disp[:, 0].unsqueeze(1)
            dl = disp[:, 1].unsqueeze(1)
            pred_imgr = get_image_from_disparity(imgl, dr)
            pred_imgl = get_image_from_disparity(imgr, -dl)

            imgl = transform(imgl)
            imgr = transform(imgr)
            dr = transform(dr)
            dl = transform(dl)
            pred_imgr = transform(pred_imgr)
            pred_imgl = transform(pred_imgl)

            imgl = imgl.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            imgr = imgr.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            dr = dr.squeeze().detach().cpu().numpy()
            dl = dl.squeeze().detach().cpu().numpy()
            pred_imgr = pred_imgr.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            pred_imgl = pred_imgl.squeeze().permute(1, 2, 0).detach().cpu().numpy()

            # base line : .54m, focal length : 721
            depth_dr = .54 * 721 / (1242 * dr)
            depth_dl = .54 * 721 / (1242 * dl)

            # depth_dr = (depth_dr - np.min(depth_dr)) / np.max(depth_dr)
            # depth_dl = (depth_dl - np.min(depth_dl)) / np.max(depth_dl)

            plt.figure(i)
            plt.subplot(421)
            plt.imshow(imgl)
            plt.subplot(422)
            plt.imshow(imgr)
            plt.subplot(423)
            plt.imshow(dr)
            plt.subplot(424)
            plt.imshow(dl)
            plt.subplot(425)
            plt.imshow(pred_imgl)
            plt.subplot(426)
            plt.imshow(pred_imgr)
            plt.subplot(427)
            plt.imshow(depth_dl, cmap='plasma', vmax=80)
            plt.subplot(428)
            plt.imshow(depth_dr, cmap='plasma', vmax=80)
            if args.show_ouptut:
                plt.show()
            if args.save_output:
                plt.savefig(args.output_dir + f'output_{i}.png')
            plt.close()
    else:
        transform = T.Compose([T.Resize((256, 512)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # transform = T.Compose([T.Resize((256, 512)), T.ToTensor()])
        transform_kitti_shape = T.Resize((375, 1242))

        for sample_pth in args.sample_pth_list:
            imgl_np = Image.open(sample_pth).convert('RGB')
            w, h = imgl_np.size
            imgl = transform(imgl_np).unsqueeze(0).to(args.device)
            disp, cont = model(imgl)[0:5:4]
            dr = disp[:, 0].unsqueeze(1)
            dl = disp[:, 1].unsqueeze(1)
            pred_imgr = get_image_from_disparity(imgl, dr)

            dr = transform_kitti_shape(dr)
            dl = transform_kitti_shape(dl)

            baseline = .54 * (w / 1242)
            focal = 721 * (w / 1242)
            depth_dr = baseline * focal / (1242 * dr)
            depth_dl = baseline * focal / (1242 * dl)

            rescale = T.Resize((h, w))
            imgl = rescale(imgl)
            dr = rescale(dr)
            dl = rescale(dl)
            depth_dr = rescale(depth_dr)
            depth_dl = rescale(depth_dl)
            pred_imgr = rescale(pred_imgr)

            imgl = imgl.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            pred_imgr = pred_imgr.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            dr = dr.squeeze().detach().cpu().numpy()
            dl = dl.squeeze().detach().cpu().numpy()
            depth_dr = depth_dr.squeeze().detach().cpu().numpy()
            depth_dl = depth_dl.squeeze().detach().cpu().numpy()
            # depth_dl = get_depth_image(depth_dl)
            # depth_dr = get_depth_image(depth_dr)

            # plt.subplot(321)
            # plt.imshow(imgl)
            # plt.subplot(322)
            # plt.imshow(pred_imgr)
            # plt.subplot(323)
            # plt.imshow(dl)
            # plt.subplot(324)
            # plt.imshow(dr)
            # plt.subplot(325)
            # plt.imshow(depth_dl, cmap='plasma', vmax=80)
            # plt.subplot(326)
            # plt.imshow(depth_dr, cmap='plasma', vmax=80)
            if args.show_contour:
                cont = cont.squeeze().detach().cpu().numpy()

                plt.subplot(211)
                plt.imshow(imgl_np)
                plt.subplot(212)
                plt.imshow(cont)
                plt.show()
            else:
                plt.subplot(311)
                plt.imshow(imgl_np)
                plt.subplot(312)
                plt.imshow(dr)
                plt.subplot(313)
                plt.imshow(dl)

            if args.show_output:
                plt.show()
            if args.save_output:
                fn = sample_pth.strip().split('/')[-1]
                plt.savefig(args.output_dir + model.__class__.__name__ + '_' + fn)
            plt.close()


if __name__ == '__main__':
    def device_type(str):
        if str == 'cuda:0' or str == 'cpu':
            return torch.device(str)
        else:
            raise argparse.ArgumentTypeError('Invalid device type')

    def bool_type(str):
        if str == 'True':
            return True
        elif str == 'False':
            return False
        else:
            raise argparse.ArgumentTypeError('Invalid bool type')

    sample_pth_list = [
        f'./samples/kitti/{i}_image_left.png' for i in range(1, 13)
    ]
    weight_pth_cdnet2 = './weights/Adam_CDNet2_kitti_50epoch_0.000010lr_1.50927loss(train)_2.00964loss_1.78435loss(ap)_0.02979loss(ds)_0.04167loss(lr)_0.15383loss(cont).ckpt'
    weight_pth_cdnet3 = './weights/Adam_CDNet3_kitti_50epoch_0.000010lr_1.50962loss(train)_1.94790loss_1.74609loss(ap)_0.02837loss(ds)_0.04166loss(lr)_0.13178loss(cont).ckpt'
    weight_pth_cdnet5 = './weights/Adam_CDNet5_kitti_50epoch_0.000010lr_1.52047loss(train)_1.94760loss_1.76012loss(ap)_0.02869loss(ds)_0.04157loss(lr)_0.11722loss(cont).ckpt'
    weight_pth_cdnet6 = './weights/Adam_CDNet6_kitti_30epoch_0.000010lr_1.60948loss(train)_1.97640loss_1.77938loss(ap)_0.02807loss(ds)_0.03899loss(lr)_0.12995loss(cont).ckpt'
    weight_pth_cdnet7 = './weights/Adam_CDNet7_kitti_50epoch_0.000100lr_1.88796loss(train)_2.06628loss_1.82601loss(ap)_0.04525loss(ds)_0.09841loss(lr)_0.09662loss(cont).ckpt'

    parser = argparse.ArgumentParser()

    parser.add_argument('--weight_pth', required=False, type=str, default=weight_pth_cdnet7)
    parser.add_argument('--sample_pth_list', required=False, nargs='+', default=sample_pth_list)
    parser.add_argument('--output_dir', required=False, type=str, default='./outputs/cdnet8/')
    parser.add_argument('--show_contour', required=False, type=bool_type, default=False)
    parser.add_argument('--show_output', required=False, type=bool_type, default=True)
    parser.add_argument('--save_output', required=False, type=bool_type, default=False)
    parser.add_argument('--device', required=False, type=device_type, default='cuda:0')

    args = parser.parse_args()

    main(args)
















