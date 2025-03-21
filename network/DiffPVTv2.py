import torch
import torch.nn as nn

from diffusion.gaussian_diffusion import get_named_beta_schedule, ModelVarType, ModelMeanType, LossType
from diffusion.resample import UniformSampler
from diffusion.respace import SpacedDiffusion, space_timesteps
from PVTv2.basic_pvt import pvt_v2_b2 as PVT
from PVTv2.basic_pvt_denoise import DPVT


class DiffPVTv2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrain = True
        self.PVT = PVT()
        self.model = DPVT()

        if self.pretrain:
            path = 'pvt_v2_b2.pth'
            save_model = torch.load(path)
            model_dict = self.PVT.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.PVT.load_state_dict(model_dict)

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


    def backbone(self, image=None):
        x0 = image
        B = x0.shape[0]

        x0, H, W = self.PVT.patch_embed1(x0)
        for i, blk in enumerate(self.PVT.block1):
            x0 = blk(x0, H, W)
        x0 = self.PVT.norm1(x0)
        x0 = x0.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x1, H, W = self.PVT.patch_embed2(x0)
        for i, blk in enumerate(self.PVT.block2):
            x1 = blk(x1, H, W)
        x1 = self.PVT.norm2(x1)
        x1 = x1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x2, H, W = self.PVT.patch_embed3(x1)
        for i, blk in enumerate(self.PVT.block3):
            x2 = blk(x2, H, W)
        x2 = self.PVT.norm3(x2)
        x2 = x2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x3, H, W = self.PVT.patch_embed4(x2)
        for i, blk in enumerate(self.PVT.block4):
            x3 = blk(x3, H, W)
        x3 = self.PVT.norm4(x3)
        x3 = x3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        return [x0, x1, x2, x3]

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
        if pred_type == "q_sample":
            # 根据标签的形状生成对应形状的噪声图像noise
            noise = torch.randn_like(x).to(x.device)
            # 从权重数组中根据概率分布进行采样，并计算相应的权重
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.backbone(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.backbone(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, 1, 256, 256),
                                                                model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out
