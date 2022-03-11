import os
import time
import datetime
import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as T

from torch.utils.data import DataLoader, ConcatDataset

from dataset.kitti_dataset import KITTIDataset, custom_collate_fn
from models.cdnet5.cdnet5 import CDNet5
from utils.pytorch_util import make_batch
from utils.util import time_calculator


def get_kitti_dataset():
    dset_name = 'kitti'
    root = 'C://DeepLearningData/KITTI/'
    shape = (256, 512)

    dset_train = KITTIDataset(
        root,
        size=shape,
        mode='train',
        normalize=False,
        use_contour=True
    )
    dset_train = ConcatDataset([dset_train for _ in range(1)])

    dset_val = KITTIDataset(
        root,
        size=shape,
        mode='val',
        normalize=False,
        use_contour=True
    )

    collate_fn = custom_collate_fn

    return dset_name, dset_train, dset_val, collate_fn


def adjust_learning_rate(optimizer, current_epoch):
    if isinstance(optimizer, optim.Adam):
        if current_epoch == 1:
            optimizer.param_groups[0]['lr'] = .0001
        elif current_epoch == 31:
            optimizer.param_groups[0]['lr'] = .00001
    elif isinstance(optimizer, optim.SGD):
        if current_epoch == 0:
            optimizer.param_groups[0]['lr'] = .001
        elif current_epoch == 30:
            optimizer.param_groups[0]['lr'] = .001
        elif current_epoch == 40:
            optimizer.param_groups[0]['lr'] = .001


def main(args):
    dset_name, dset_train, dset_val, collate_fn = get_kitti_dataset()

    model = CDNet5().to(args.device)
    model_name = model.__class__.__name__

    try:
        if args.optimizer == 'Adam' or 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'SGD' or 'sgd' or 'Sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            raise Exception('Invalid optimizer name.')
    except Exception:
        pass
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    finally:
        optim_name = optimizer.__class__.__name__

    if len(args.ckpt_pth) > 0:
        ckpt = torch.load(args.ckpt_pth)
        model.load_state_dict(ckpt['model_state_dict'])
        if optim_name == 'Adam' and 'optimizer_state_dict' in ckpt.keys():
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    if not args.debug and args.save_record:
        now = datetime.datetime.now()
        date_str = f'{now.date().year}{now.date().month:02d}{now.date().day:02d}'
        time_str = f'{now.time().hour:02d}{now.time().minute:02d}{now.time().second:02d}'
        record_dir = os.path.join('./records', model_name)
        if not os.path.exists(record_dir):
            os.mkdir(record_dir)
        record_fn = f'record_{model_name}_{date_str}_{time_str}.csv'
        record_pth = os.path.join(record_dir, record_fn)
        with open(record_pth, 'w') as record:
            record.write('optimizer, epoch, lr, loss(train), loss, loss(ap), loss(ds), loss(lr), loss(cont)\n')

    train_loss_list = []
    val_loss_list = []

    t_start = time.time()
    for e in range(args.epoch):
        num_batch = 0
        num_data = 0
        train_loss = 0
        train_loss_ap = 0
        train_loss_ds = 0
        train_loss_lr = 0
        train_loss_cont = 0

        adjust_learning_rate(optimizer, e+1)
        cur_lr = optimizer.param_groups[0]['lr']
        train_loader = DataLoader(dataset=dset_train, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=custom_collate_fn, num_workers=1)
        t_train_start = time.time()
        model.train()
        for i, ann in enumerate(train_loader):
            num_batch += 1
            num_data += len(ann)

            print(f'{optim_name} ', end='')
            print(f'[{e+1}/{args.epoch}] ', end='')
            print(f'{num_data}/{len(dset_train)}  ', end='')
            print(f'<lr> {cur_lr:.6f}  ', end='')

            imgl = make_batch(ann, 'imgl').to(args.device)
            imgr = make_batch(ann, 'imgr').to(args.device)
            tar_cont = make_batch(ann, 'contour').to(args.device)

            disp1, disp2, disp3, disp4, pred_cont = model(imgl)
            optimizer.zero_grad()
            loss, loss_ap, loss_ds, loss_lr, loss_cont = model.loss(imgl, imgr, [disp1, disp2, disp3, disp4], pred_cont, tar_cont)
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().cpu().item()
            train_loss_ap += loss_ap
            train_loss_ds += loss_ds
            train_loss_lr += loss_lr
            train_loss_cont += loss_cont

            t_batch_end = time.time()
            h, m, s = time_calculator(t_batch_end - t_start)

            print(f'<loss> {loss.detach().cpu().item():.5f} ({train_loss/num_batch:.5f})  ', end='')
            print(f'<loss_ap> {loss_ap:.5f} ({train_loss_ap/num_batch:.5f})  ', end='')
            print(f'<loss_ds> {loss_ds:.5f} ({train_loss_ds/num_batch:.5f})  ', end='')
            print(f'<loss_lr> {loss_lr:.5f} ({train_loss_lr/num_batch:.5f})  ', end='')
            print(f'<loss_cont> {loss_cont:.5f} ({train_loss_cont/num_batch:.5f})  ', end='')
            print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}', end='')

            del imgl, imgr, disp1, disp2, disp3, disp4, loss, loss_ap, loss_ds, loss_lr, loss_cont
            torch.cuda.empty_cache()

            if args.debug:
                if num_data == args.batch_size * 5:
                    print()
                else:
                    print(end='\r')
                if i == 4:
                    break
            else:
                if num_data == len(dset_train):
                    print()
                else:
                    print(end='\r')

        del train_loader

        t_train_end = time.time()
        h, m, s = time_calculator(t_train_end - t_train_start)

        train_loss /= num_batch
        train_loss_ap /= num_batch
        train_loss_ds /= num_batch
        train_loss_lr /= num_batch
        train_loss_cont /= num_batch

        train_loss_list.append(train_loss)

        print('\t\t(train) - ', end='')
        print(f'<loss> {train_loss:.5f}  ', end='')
        print(f'<loss_ap> {train_loss_ap:.5f}  ', end='')
        print(f'<loss_ds> {train_loss_ds:.5f}  ', end='')
        print(f'<loss_lr> {train_loss_lr:.5f}  ', end='')
        print(f'<loss_cont> {train_loss_cont:.5f}  ', end='')
        print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}')

        num_batch = 0
        num_data = 0
        val_loss = 0
        val_loss_ap = 0
        val_loss_ds = 0
        val_loss_lr = 0
        val_loss_cont = 0
        val_loader = DataLoader(dataset=dset_val, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=custom_collate_fn, num_workers=1)
        t_val_start = time.time()
        model.eval()
        for i, ann in enumerate(val_loader):
            num_batch += 1
            num_data += len(ann)

            imgl = make_batch(ann, 'imgl').to(args.device)
            imgr = make_batch(ann, 'imgr').to(args.device)
            tar_cont = make_batch(ann, 'contour').to(args.device)

            disp1, disp2, disp3, disp4, pred_cont = model(imgl)
            loss, loss_ap, loss_ds, loss_lr, loss_cont = model.loss(imgl, imgr, [disp1, disp2, disp3, disp4], pred_cont, tar_cont)

            val_loss += loss.detach().cpu().item()
            val_loss_ap += loss_ap
            val_loss_ds += loss_ds
            val_loss_lr += loss_lr
            val_loss_cont += loss_cont

            print(f'Validating {num_data}/{len(dset_val)}', end='\r')

            del imgl, imgr, disp1, disp2, disp3, disp4, loss, loss_ap, loss_ds, loss_lr, loss_cont
            torch.cuda.empty_cache()

            if args.debug and i == 4:
                break

        del val_loader

        t_val_end = time.time()
        h, m, s = time_calculator(t_val_end - t_val_start)

        val_loss /= num_batch
        val_loss_ap /= num_batch
        val_loss_ds /= num_batch
        val_loss_lr /= num_batch
        val_loss_cont /= num_batch

        val_loss_list.append(val_loss)

        print('\t\t(valid) - ', end='')
        print(f'<loss> {val_loss:.5f}  ', end='')
        print(f'<loss_ap> {val_loss_ap:.5f}  ', end='')
        print(f'<loss_ds> {val_loss_ds:.5f}  ', end='')
        print(f'<loss_lr> {val_loss_lr:.5f}  ', end='')
        print(f'<loss_cont> {val_loss_cont:.5f}  ', end='')
        print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}')

        if not args.debug and args.save_record:
            with open(record_pth, 'a') as record:
                record_str = f'{optim_name}, {e+1}, {cur_lr}, {train_loss:.5f}, {val_loss:.5f}, {val_loss_ap:.5f}, {val_loss_ds:.5f}, {val_loss_lr:.5f}, {val_loss_cont:.5f}\n'
                record.write(record_str)

        if not args.debug and (e + 1) % 1 == 0:
            save_pth = f'./save/(2nd){optim_name}_{model_name}_{dset_name}_{e+1}epoch_{cur_lr:.6f}lr_{train_loss:.5f}loss(train)_{val_loss:.5f}loss_{val_loss_ap:.5f}loss(ap)_{val_loss_ds:.5f}loss(ds)_{val_loss_lr:.5f}loss(lr)_{val_loss_cont:.5f}loss(cont).ckpt'
            save_dict = {'model_state_dict': model.state_dict()}
            if optim_name == 'Adam':
                save_dict['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(save_dict, save_pth)

        x_axis = [i for i in range(len(train_loss_list))]
        plt.figure(e)
        plt.plot(x_axis, train_loss_list, 'r-', label='train')
        plt.plot(x_axis, val_loss_list, 'b-', label='val')
        plt.title('Loss')
        plt.legend()
        if not args.debug and args.save_graph:
            plt.savefig(f'./graph/(2nd)loss_{model_name}_{date_str}_{time_str}.png')
        plt.close()


if __name__ == '__main__':
    def type_bool(value):
        if value == 'True':
            return True
        elif value == 'False':
            return False
        else:
            argparse.ArgumentTypeError('Not Boolean value.')

    ckpt_pth = ''

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', required=False, type=str, default='cuda:0')
    parser.add_argument('--batch_size', required=False, type=int, default=2)
    parser.add_argument('--lr', required=False, type=float, default=.0001)
    parser.add_argument('--momentum', required=False, type=float, default=.9)
    parser.add_argument('--weight_decay', required=False, type=float, default=.0005)
    parser.add_argument('--epoch', required=False, type=int, default=50)
    parser.add_argument('--save_graph', required=False, type=type_bool, default=True)
    parser.add_argument('--save_record', required=False, type=type_bool, default=True)
    parser.add_argument('--optimizer', required=False, type=str, default='Adam')
    parser.add_argument('--ckpt_pth', required=False, type=str, default=ckpt_pth)    # Blank if none.

    parser.add_argument('--debug', required=False, type=type_bool, default=False)

    args = parser.parse_args()

    torch.set_num_threads(1)

    main(args)
