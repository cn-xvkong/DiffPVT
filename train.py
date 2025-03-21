import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import argparse
from monai.losses.dice import DiceLoss
from torch.utils.data import DataLoader

from network.DiffDualBranch import DiffDualBranch
from network.DiffPVTv2 import DiffPVTv2
from utils.metric import jac_score, dice_score, precision, recall

from network.DiffUNet import DiffUNet
from network.DiffRes2Net import DiffRes2Net
from utils.dataloader_r import get_loader


def getArgs():
    parse = argparse.ArgumentParser()
    # 优化器、学习率参数
    parse.add_argument('--learning_rate', type=int, default=1e-4)
    parse.add_argument('--weight_decay', type=int, default=1e-3)
    # 数据集
    parse.add_argument('--dataset', type=str, default="ISIC2018")
    parse.add_argument('--data_dir', type=str, default="./Dataset/ISIC2018")
    # 训练参数
    parse.add_argument('--val_every', type=int, default=10)  # 每隔*轮验证
    parse.add_argument('--max_epoch', type=int, default=2000)  # 训练轮数
    parse.add_argument('--batch_size', type=int, default=1)  # 批处理图像大小
    parse.add_argument('--device_ids', type=str, default="0")  # 使用GPU *
    parse.add_argument('--logdir', type=str, default="models/DUN")  # 模型保存
    parse.add_argument('--net', type=str, default="DUN", help="DUN,DRN,DTN,DDBN")  # 网络选择
    args = parse.parse_args()

    return args


def train(args, model, train_load, test_load, logdir, max_epoch, device, scheduler):
    print("==============================================")
    print(f"CUDA ID: {next(model.parameters()).device}")
    print(f"Model's name: {args.net}")
    print(f"Batch size: {args.batch_size}")
    print(f"Check model's parameter: {next(model.parameters()).sum()}")
    print(f"Model's parameter: {sum([np.prod(list(p.size())) for p in model.parameters()]) * 4 / 1000 / 1000}M ")
    print("==============================================")
    print("start the train process")
    print("==============================================")
    print()

    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)

    best_f1 = 0.0
    os.makedirs(logdir, exist_ok=True)
    for epoch in range(0, max_epoch):
        total_loss = 0.0
        model.train()
        with tqdm(total=len(train_load)) as k:
            for idx, batch in enumerate(train_load):
                image = batch["image"].to(device)
                label = batch["mask"].float().to(device)

                # pred_type = "q_sample" / "denoise" / "ddim_sample"
                x_start = label
                x_start = (x_start) * 2 - 1

                # 对标签加入噪声的过程
                x_t, t, noise = model(x=x_start, pred_type="q_sample")
                # 去噪过程
                pred_start = model(x=x_t, step=t, image=image, pred_type="denoise")

                loss_dice = dice_loss(pred_start, label)
                loss_bce = bce(pred_start, label)

                pred_start = torch.sigmoid(pred_start)
                loss_mse = mse(pred_start, label)

                batch_loss = loss_dice + loss_bce + loss_mse

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                k.set_postfix(loss=total_loss, lr=lr, epoch=epoch)
                k.update(1)
        if epoch == 0:
            mF1 = 0
        if epoch % args.val_every == 0 and epoch != 0:
            model.eval()
            dices = []
            iou = []
            rec = []
            prec = []
            for idx_v, batch_v in tqdm(enumerate(test_load), total=len(test_load)):
                image = batch_v["image"].to(device)
                label = batch_v["mask"].float().to(device)

                with torch.no_grad():
                    # 验证扩散去噪网络
                    output = model(image=image, pred_type="ddim_sample")
                    output = torch.sigmoid(output)

                output = output.cpu().numpy() > 0.5
                target = label.cpu().numpy()

                dices.append(dice_score(target, output))
                rec.append(recall(target, output))
                prec.append(precision(target, output))
                iou.append(jac_score(target, output))

                mF1 = np.mean(dices)
                mIoU = np.mean(iou)
                mRecall = np.mean(rec)
                mPrecision = np.mean(prec)

            print(f"mF1={mF1}", f"miou={mIoU}", f"recall={mRecall}", f"Precision={mPrecision}")
        scheduler.step()  # 调整学习率
        if args.dataset == "ISIC2016":
            if mF1 > best_f1:
                best_f1 = mF1
                torch.save(model.state_dict(), logdir + "/ISIC2016_best_model.pth")
                print("***************************************************")
                print("*****************best_model has saved**************")
                print("***************************************************")
            torch.save(model.state_dict(), logdir + "/ISIC2016_last_model.pth")
            print("*****************last model has saved**************")
        elif args.dataset == "ISIC2017":
            if mF1 > best_f1:
                best_f1 = mF1
                torch.save(model.state_dict(), logdir + "/ISIC2017_best_model.pth")
                print("***************************************************")
                print("*****************best_model has saved**************")
                print("***************************************************")
            torch.save(model.state_dict(), logdir + "/ISIC2017_last_model.pth")
            print("*****************last model has saved**************")
        elif args.dataset == "ISIC2018":
            if mF1 > best_f1:
                best_f1 = mF1
                torch.save(model.state_dict(), logdir + "/ISIC2018_best_model.pth")
                print("***************************************************")
                print("*****************best_model has saved**************")
                print("***************************************************")
            torch.save(model.state_dict(), logdir + "/ISIC2018_last_model.pth")
            print("*****************last model has saved**************")
        elif args.dataset == "PH2":
            if mF1 > best_f1:
                best_f1 = mF1
                torch.save(model.state_dict(), logdir + "/PH2_best_model.pth")
                print("***************************************************")
                print("*****************best_model has saved**************")
                print("***************************************************")
            torch.save(model.state_dict(), logdir + "/PH2_last_model.pth")
            print("*****************last model has saved**************")
        elif args.dataset == "GLaS":
            if mF1 > best_f1:
                best_f1 = mF1
                torch.save(model.state_dict(), logdir + "/GLaS_best_model.pth")
                print("***************************************************")
                print("*****************best_model has saved**************")
                print("***************************************************")
            torch.save(model.state_dict(), logdir + "/GLaS_last_model.pth")
            print("*****************last model has saved**************")
        elif args.dataset == "DSB2":
            if mF1 > best_f1:
                best_f1 = mF1
                torch.save(model.state_dict(), logdir + "/DSB2_best_model.pth")
                print("***************************************************")
                print("*****************best_model has saved**************")
                print("***************************************************")
            torch.save(model.state_dict(), logdir + "/DSB2_last_model.pth")
            print("*****************last model has saved**************")
        elif args.dataset == "SEG":
            if mF1 > best_f1:
                best_f1 = mF1
                torch.save(model.state_dict(), logdir + "/SEG_best_model.pth")
                print("***************************************************")
                print("*****************best_model has saved**************")
                print("***************************************************")
            torch.save(model.state_dict(), logdir + "/SEG_last_model.pth")
            print("*****************last model has saved**************")
        elif args.dataset == "Sess":
            if mF1 > best_f1:
                best_f1 = mF1
                torch.save(model.state_dict(), logdir + "/Sess_best_model.pth")
                print("***************************************************")
                print("*****************best_model has saved**************")
                print("***************************************************")
            torch.save(model.state_dict(), logdir + "/Sess_last_model.pth")
            print("*****************last model has saved**************")
        else:
            print("Dataset error!")
            exit(0)
    return 0


if __name__ == "__main__":
    # 初始化参数
    args = getArgs()

    device_ids = [int(i) for i in args.device_ids.split(',')]
    device = torch.device(f'cuda:{device_ids[0]}')

    train_ds, val_ds, test_ds = get_loader(data_dir=args.data_dir)
    train_load = DataLoader(dataset=train_ds, shuffle=True, batch_size=args.batch_size, drop_last=True)
    test_load = DataLoader(dataset=test_ds, shuffle=True, batch_size=1, drop_last=True)

    if args.net == "DUN":
        model = DiffUNet()
    elif args.net == "DRN":
        model = DiffRes2Net()
    elif args.net == "DTN":
        model = DiffPVTv2()
    elif args.net == "DDBN":
        model = DiffDualBranch()
    else:
        print("Net is error")
        exit(0)

    if torch.cuda.is_available():
        model.to(device)

    # 损失函数
    mse = nn.MSELoss()
    dice_loss = DiceLoss(sigmoid=True)
    bce = nn.BCEWithLogitsLoss()
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # 学习率
    # scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)

    train(args=args, model=model, train_load=train_load, test_load=test_load, logdir=args.logdir,
          max_epoch=args.max_epoch, device=device, scheduler=scheduler)
