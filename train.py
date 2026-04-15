import os, time, math
import numpy as np

import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.backends import cudnn
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from logger import plot_loss_log, plot_psnr_log
from metric import psnr, ssim
from model.backbone_train import DEANet
from loss import ContrastLoss
from option_train import opt
from data.data_loader import TrainDataset, TestDataset

start_time = time.time()
steps = opt.iters_per_epoch * opt.epochs
T = steps

def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr

def train(net, loader_train, loader_test, optimizer, criterion):
    losses = []
    loss_log = {'L1': [], 'CR': [], 'total': []}
    loss_log_tmp = {'L1': [], 'CR': [], 'total': []}
    psnr_log = []

    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    loader_train_iter = iter(loader_train)

    for step in range(start_step + 1, steps + 1):
        net.train()
        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        # 获取训练数据
        try:
            x, y = next(loader_train_iter)
        except StopIteration:
            loader_train_iter = iter(loader_train)
            x, y = next(loader_train_iter)

        x = x.to(opt.device)
        y = y.to(opt.device)

        out = net(x)
        if out.size(2) != y.size(2) or out.size(3) != y.size(3):
            min_h = min(out.size(2), y.size(2))
            min_w = min(out.size(3), y.size(3))
            out = out[:, :, :min_h, :min_w]
            y = y[:, :, :min_h, :min_w]

        if opt.w_loss_L1 > 0:
            loss_L1 = criterion[0](out, y)
        if opt.w_loss_CR > 0:
            loss_CR = criterion[1](out, y, x)
        loss = opt.w_loss_L1 * loss_L1 + opt.w_loss_CR * loss_CR
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.detach().item())  # 添加 detach()，避免损失带有梯度信息

        loss_log_tmp['L1'].append(loss_L1.item())
        loss_log_tmp['CR'].append(loss_CR.item())
        loss_log_tmp['total'].append(loss.item())

        print(
            f'\rloss:{loss.item():.5f} | L1:{loss_L1.item():.5f} | CR:{opt.w_loss_CR * loss_CR.item():.5f} | step:{step}/{steps} | lr:{lr:.7f} | time_used:{(time.time()-start_time)/60:.1f}min',
            end='', flush=True)

        if step % len(loader_train) == 0:
            loader_train_iter = iter(loader_train)
            for key in loss_log.keys():
                loss_log[key].append(np.average(np.array(loss_log_tmp[key])))
                loss_log_tmp[key] = []
            plot_loss_log(loss_log, int(step / len(loader_train)), opt.saved_plot_dir)
            # 确保保存损失目录存在
            if not os.path.exists(opt.saved_data_dir):
                os.makedirs(opt.saved_data_dir)
            np.save(os.path.join(opt.saved_data_dir, 'losses.npy'), losses)

        if (step % opt.iters_per_epoch == 0 and step <= opt.finer_eval_step) or \
           (step > opt.finer_eval_step and (step - opt.finer_eval_step) % (5 * len(loader_train)) == 0):
            if step > opt.finer_eval_step:
                epoch = opt.finer_eval_step // opt.iters_per_epoch + (step - opt.finer_eval_step) // (5 * len(loader_train))
            else:
                epoch = int(step / opt.iters_per_epoch)
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test)

            log = f'\nstep:{step} | epoch:{epoch} | ssim:{ssim_eval:.4f} | psnr:{psnr_eval:.4f}'
            print(log)
            with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                f.write(log + '\n')

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            psnr_log.append(psnr_eval)
            plot_psnr_log(psnr_log, epoch, opt.saved_plot_dir)

            # 当 psnr 提升时，保存当前最优模型（仅保存模型参数）
            if psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                print(f'\nModel saved at step:{step} | epoch:{epoch} | max_psnr:{max_psnr:.4f} | max_ssim:{max_ssim:.4f}')
                saved_best_model_path = os.path.join(opt.saved_model_dir, 'best.pth')
                if isinstance(net, torch.nn.DataParallel):
                    torch.save(net.module.state_dict(), saved_best_model_path)
                else:
                    torch.save(net.state_dict(), saved_best_model_path)
            # 每个 epoch 单独保存一次模型（仅保存模型参数）
            saved_single_model_path = os.path.join(opt.saved_model_dir, f'{epoch}.pth')
            if isinstance(net, torch.nn.DataParallel):
                torch.save(net.module.state_dict(), saved_single_model_path)
            else:
                torch.save(net.state_dict(), saved_single_model_path)
            loader_train_iter = iter(loader_train)
            np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
            np.save(os.path.join(opt.saved_data_dir, 'psnrs.npy'), psnrs)

def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def test(net, loader_test):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

    for i, (inputs, targets, hazy_name) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        with torch.no_grad():
            H, W = inputs.shape[2:]
            inputs = pad_img(inputs, 4)
            pred = net(inputs).clamp(0, 1)
            pred = pred[:, :, :H, :W]
            save_path = os.path.join(opt.saved_infer_dir, hazy_name[0])
            save_image(pred, save_path)
        ssim_tmp = ssim(pred, targets).item()
        psnr_tmp = psnr(pred, targets)
        ssims.append(ssim_tmp)
        psnrs.append(psnr_tmp)

    return np.mean(ssims), np.mean(psnrs)

def set_seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    set_seed_torch(666)
    torch.cuda.empty_cache()

    train_dir = '../dataset/LSUI/train'
    train_set = TrainDataset(os.path.join(train_dir, 'hazy'), os.path.join(train_dir, 'clear'))
    test_dir = '../dataset/LSUI/test'
    test_set = TestDataset(os.path.join(test_dir, 'hazy'), os.path.join(test_dir, 'clear'))
    loader_train = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=0)
    loader_test = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0)

    net = DEANet(opt,base_dim=32)
    net = net.to(opt.device)

    epoch_size = len(loader_train)
    print("epoch_size:", epoch_size)
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))

    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    criterion.append(ContrastLoss(ablation=False))

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()),
                           lr=opt.start_lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer.zero_grad()

    # 如果存在预训练模型，则加载检查点（只加载模型参数）
    if opt.pre_trained_model != 'null' and os.path.exists(opt.pre_trained_model):
        print(f"Resuming training from checkpoint: {opt.pre_trained_model}")
        checkpoint = torch.load(opt.pre_trained_model)
        # 如果加载的是完整检查点，则提取 'model' 部分
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        net.load_state_dict(checkpoint, strict=False)
    train(net, loader_train, loader_test, optimizer, criterion)
