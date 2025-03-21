import os
import torch
import argparse
import matplotlib
import matplotlib.pyplot as plt
from torch import nn
from collections import OrderedDict

from network.DiffDualBranch import DiffDualBranch
from network.DiffPVTv2 import DiffPVTv2
from network.DiffRes2Net import DiffRes2Net
from utils.dataloader_r import *
from torch.utils.data import DataLoader

from utils.metric import jac_score, dice_score, precision, recall

from network.DiffUNet import DiffUNet

matplotlib.use('Agg')

def getArgs():
    parse = argparse.ArgumentParser()
    # 数据集
    parse.add_argument('--test_data_dir', type=str, default="./Dataset/ISIC2018")
    # 测试参数
    parse.add_argument('--batch_size', type=int, default=1)
    parse.add_argument('--device_ids', type=str, default="6")
    parse.add_argument('--result', type=str, default="Predictions/DTN/ISIC2018")
    parse.add_argument('--model_path', type=str, default="models/DTN/ISIC2018_best_model.pth")
    parse.add_argument('--net', type=str, default="DTN", help="DUN,DRN,DTN,DDBN")
    args = parse.parse_args()

    return args


def test_save(args, model, test_load, device):
    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    mIoU = 0.0
    mF1 = 0.0
    mRecall = 0.0
    mPrecision = 0.0
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_load), total=len(test_load)):
            image = batch["image"].to(device)
            label = batch["mask"].float().to(device)
            path = batch["path"]

            # 测试扩散去噪网络
            output = model(image=image, pred_type="ddim_sample")
            output = torch.sigmoid(output)

            output = output.cpu().numpy() > 0.5

            # 计算指标
            target = label.cpu().numpy()

            mIoU += jac_score(target, output)
            mF1 += dice_score(target, output)
            mRecall += recall(target, output)
            mPrecision += precision(target, output)

            save_img = output[0][0]
            # 保存预测结果
            os.makedirs(args.result, exist_ok=True)
            plt.imshow(save_img, cmap="gray")
            plt.imsave(args.result + '/' + os.path.basename(path[0]).split('/')[-1][:-4] + '.png', save_img,
                       cmap="gray")

        # 计算平均指标
        num_samples = len(test_load.dataset)
        mIoU /= num_samples
        mF1 /= num_samples
        mRecall /= num_samples
        mPrecision /= num_samples

    # 打印, 保存指标
    print(f"mF1={mF1}", f"mIoU={mIoU}", f"Recall={mRecall}", f"Precision={mPrecision}")

    return 0


if __name__ == '__main__':
    args = getArgs()
    print("Loading model...")

    device_ids = [int(i) for i in args.device_ids.split(',')]
    device = torch.device(f'cuda:{device_ids[0]}')

    if args.net == "DUN":
        model = DiffUNet()
        args.model_path = args.model_path
    elif args.net == "DRN":
        model = DiffRes2Net()
        args.model_path = args.model_path
    elif args.net == "DTN":
        model = DiffPVTv2()
        args.model_path = args.model_path
    elif args.net == "DDBN":
        model = DiffDualBranch()
        args.model_path = args.model_path
    else:
        print("Net is error")
        exit(0)

    new_state_dict = OrderedDict()
    model_state_dict=torch.load(args.model_path, map_location='cuda:6')
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith("module.") else k 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)  # 模型文件路径
    print(f"Sucessfully load weight from {args.model_path}")

    model.to(device)
    model.eval()

    args.result = args.result
    os.makedirs(args.result, exist_ok=True)

    # 测试数据集
    _, _, test_ds = get_loader(args.test_data_dir, cache=False)
    test_load = DataLoader(dataset=test_ds, shuffle=True, batch_size=1, drop_last=True)
    test_save(args=args, model=model, test_load=test_load, device=device)


# import matplotlib
# import matplotlib.pyplot as plt
#
# matplotlib.use('TkAgg')
#
# x0 = x0.squeeze().cpu().detach()
# plt.imshow(x0, cmap="gray")
# plt.show()
