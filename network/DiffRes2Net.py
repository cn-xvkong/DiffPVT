import torch
import torch.nn as nn
import timm

from diffusion.gaussian_diffusion import get_named_beta_schedule, ModelVarType, ModelMeanType, LossType
from diffusion.resample import UniformSampler
from diffusion.respace import SpacedDiffusion, space_timesteps
from Res2Net.basic_res2net import res2net50_26w_4s as Res2Net
from Res2Net.basic_res2net import DRes2Net


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DiffRes2Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrain = True
        self.drop_rate = 0.2
        self.Res2Net = Res2Net()
        self.model = DRes2Net()

        if self.pretrain:
            pre_path = 'res2net50_26w_4s-06e79181.pth'
            save_model = timm.create_model('res2net50_26w_4s', pretrained=True, pretrained_cfg=dict(file=pre_path))
            pretrained_state_dict = save_model.state_dict()
            self.Res2Net.load_state_dict(pretrained_state_dict)
        self.Res2Net.layer4 = nn.Identity()

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(1000)

        self.drop = nn.Dropout2d(self.drop_rate)

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            x0 = self.Res2Net.conv1(image)
            x0 = self.Res2Net.bn1(x0)
            x0 = self.Res2Net.relu(x0)

            x1 = self.Res2Net.maxpool(x0)
            x1 = self.Res2Net.layer1(x1)
            x1 = self.drop(x1)

            x2 = self.Res2Net.layer2(x1)
            x2 = self.drop(x2)

            x3 = self.Res2Net.layer3(x2)
            x3 = self.drop(x3)

            x4 = self.Res2Net.maxpool(x3)
            x4 = self.Res2Net.layer4(x4)

            embeddings = [x0, x1, x2, x3, x4]
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            x0 = self.Res2Net.conv1(image)
            x0 = self.Res2Net.bn1(x0)
            x0 = self.Res2Net.relu(x0)

            x1 = self.Res2Net.maxpool(x0)
            x1 = self.Res2Net.layer1(x1)
            x1 = self.drop(x1)

            x2 = self.Res2Net.layer2(x1)
            x2 = self.drop(x2)

            x3 = self.Res2Net.layer3(x2)
            x3 = self.drop(x3)

            x4 = self.Res2Net.maxpool(x3)
            x4 = self.Res2Net.layer4(x4)

            embeddings = [x0, x1, x2, x3, x4]
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, 1, 256, 256),
                                                                model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out
