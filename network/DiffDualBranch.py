import torch
import torch.nn as nn
import timm

from diffusion.gaussian_diffusion import get_named_beta_schedule, ModelVarType, ModelMeanType, LossType
from diffusion.resample import UniformSampler
from diffusion.respace import SpacedDiffusion, space_timesteps
from Res2Net.basic_res2net import res2net50_26w_4s as Res2Net
from PVTv2.basic_pvt import pvt_v2_b2 as PVT
from DualBranch.RDFM import RDFM
from DualBranch.basic_dualbranch import Denoise_DualBranch


class DiffDualBranch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pretrain = True
        self.drop_rate = 0.2
        self.Res2Net = Res2Net()
        self.PVT = PVT()
        self.model = Denoise_DualBranch()

        if self.pretrain:
            pre_path = 'res2net50_26w_4s-06e79181.pth'
            save_model = timm.create_model('res2net50_26w_4s', pretrained=True, pretrained_cfg=dict(file=pre_path))
            pretrained_state_dict = save_model.state_dict()
            self.Res2Net.load_state_dict(pretrained_state_dict)
        self.Res2Net.layer4 = nn.Identity()

        if self.pretrain:
            path = 'pvt_v2_b2.pth'
            save_model = torch.load(path)
            model_dict = self.PVT.state_dict()
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.PVT.load_state_dict(model_dict)

        self.RDFM1 = RDFM(in_channel_high=1024, in_channel_low=512, ratio=4, out_channel=512, drop_rate=self.drop_rate)
        self.RDFM2 = RDFM(in_channel_high=1024, in_channel_low=320, ratio=4, out_channel=320, drop_rate=self.drop_rate)
        self.RDFM3 = RDFM(in_channel_high=512, in_channel_low=128, ratio=4, out_channel=128, drop_rate=self.drop_rate)
        self.RDFM4 = RDFM(in_channel_high=256, in_channel_low=64, ratio=4, out_channel=64, drop_rate=self.drop_rate)

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

            # CNN Branch
            r0 = self.Res2Net.conv1(image)
            r0 = self.Res2Net.bn1(r0)
            r0 = self.Res2Net.relu(r0)

            r1 = self.Res2Net.maxpool(r0)
            r1 = self.Res2Net.layer1(r1)
            r1 = self.drop(r1)

            r2 = self.Res2Net.layer2(r1)
            r2 = self.drop(r2)

            r3 = self.Res2Net.layer3(r2)
            r3 = self.drop(r3)

            r4 = self.Res2Net.maxpool(r3)
            r4 = self.Res2Net.layer4(r4)

            # Transformer Branch
            p0 = image
            B = p0.shape[0]

            p0, H, W = self.PVT.patch_embed1(p0)
            for i, blk in enumerate(self.PVT.block1):
                p0 = blk(p0, H, W)
            p0 = self.PVT.norm1(p0)
            p0 = p0.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            p1, H, W = self.PVT.patch_embed2(p0)
            for i, blk in enumerate(self.PVT.block2):
                p1 = blk(p1, H, W)
            p1 = self.PVT.norm2(p1)
            p1 = p1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            p2, H, W = self.PVT.patch_embed3(p1)
            for i, blk in enumerate(self.PVT.block3):
                p2 = blk(p2, H, W)
            p2 = self.PVT.norm3(p2)
            p2 = p2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            p3, H, W = self.PVT.patch_embed4(p2)
            for i, blk in enumerate(self.PVT.block4):
                p3 = blk(p3, H, W)
            p3 = self.PVT.norm4(p3)
            p3 = p3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            fusion4 = self.RDFM1(r4, p3)
            fusion3 = self.RDFM2(r3, p2)
            fusion2 = self.RDFM3(r2, p1)
            fusion1 = self.RDFM4(r1, p0)

            embeddings = [fusion1, fusion2, fusion3, fusion4]
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":

            # CNN Branch
            r0 = self.Res2Net.conv1(image)
            r0 = self.Res2Net.bn1(r0)
            r0 = self.Res2Net.relu(r0)

            r1 = self.Res2Net.maxpool(r0)
            r1 = self.Res2Net.layer1(r1)
            r1 = self.drop(r1)

            r2 = self.Res2Net.layer2(r1)
            r2 = self.drop(r2)

            r3 = self.Res2Net.layer3(r2)
            r3 = self.drop(r3)

            r4 = self.Res2Net.maxpool(r3)
            r4 = self.Res2Net.layer4(r4)

            # Transformer Branch
            p0 = image
            B = p0.shape[0]

            p0, H, W = self.PVT.patch_embed1(p0)
            for i, blk in enumerate(self.PVT.block1):
                p0 = blk(p0, H, W)
            p0 = self.PVT.norm1(p0)
            p0 = p0.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            p1, H, W = self.PVT.patch_embed2(p0)
            for i, blk in enumerate(self.PVT.block2):
                p1 = blk(p1, H, W)
            p1 = self.PVT.norm2(p1)
            p1 = p1.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            p2, H, W = self.PVT.patch_embed3(p1)
            for i, blk in enumerate(self.PVT.block3):
                p2 = blk(p2, H, W)
            p2 = self.PVT.norm3(p2)
            p2 = p2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            p3, H, W = self.PVT.patch_embed4(p2)
            for i, blk in enumerate(self.PVT.block4):
                p3 = blk(p3, H, W)
            p3 = self.PVT.norm4(p3)
            p3 = p3.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            fusion4 = self.RDFM1(r4, p3)
            fusion3 = self.RDFM2(r3, p2)
            fusion2 = self.RDFM3(r2, p1)
            fusion1 = self.RDFM4(r1, p0)

            embeddings = [fusion1, fusion2, fusion3, fusion4]
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, 1, 256, 256),
                                                                model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out
