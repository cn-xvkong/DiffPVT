import torch
import torch.nn as nn

from diffusion.gaussian_diffusion import get_named_beta_schedule, ModelVarType, ModelMeanType, LossType
from diffusion.resample import UniformSampler
from diffusion.respace import SpacedDiffusion, space_timesteps
from unet.basic_unet import BasicUNetEncoder
from unet.basic_unet_denose import BasicUNetDe


class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(
            spatial_dims=2,
            in_channels=3,
            out_channels=2,
            features=[64, 64, 128, 256, 512, 64]
        )

        self.model = BasicUNetDe(
            spatial_dims=2,
            in_channels=4,
            out_channels=1,
            features=[64, 64, 128, 256, 512, 64],
            act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False})
        )

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

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, 1, 256, 256),
                                                                model_kwargs={"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out
