import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from diffusers import DDPMScheduler
import math
from torchvision.models._utils import IntermediateLayerGetter
import torchvision

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

class DiffusionSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionConv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class DiffusionConditionalResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels)
        )
        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.conv1(x)
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        out = out + cond_embed
        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out

class DiffusionUnet(nn.Module):
    def __init__(self, action_dim, global_cond_dim, down_dims=[256, 512, 1024], kernel_size=5):
        super().__init__()
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(128),
            nn.Linear(128, 512),
            nn.Mish(),
            nn.Linear(512, 128),
        )
        cond_dim = 128 + global_cond_dim
        in_out = [(action_dim, down_dims[0])] + list(zip(down_dims[:-1], down_dims[1:]))
        self.down_modules = nn.ModuleList([
            nn.ModuleList([
                DiffusionConditionalResidualBlock1d(dim_in, dim_out, cond_dim, kernel_size),
                DiffusionConditionalResidualBlock1d(dim_out, dim_out, cond_dim, kernel_size),
                nn.Conv1d(dim_out, dim_out, 3, 2, 1) if dim_out != down_dims[-1] else nn.Identity(),
            ]) for dim_in, dim_out in in_out
        ])
        self.mid_modules = nn.ModuleList([
            DiffusionConditionalResidualBlock1d(down_dims[-1], down_dims[-1], cond_dim, kernel_size),
            DiffusionConditionalResidualBlock1d(down_dims[-1], down_dims[-1], cond_dim, kernel_size),
        ])
        self.up_modules = nn.ModuleList([
            nn.ModuleList([
                DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, cond_dim, kernel_size),
                DiffusionConditionalResidualBlock1d(dim_out, dim_out, cond_dim, kernel_size),
                nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1),  # Always upsample
            ]) for dim_out, dim_in in reversed(list(zip(down_dims[:-1], down_dims[1:])))
        ])
        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(down_dims[0], down_dims[0], kernel_size),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(self, x, timestep, global_cond):
        x = x.permute(0, 2, 1)  # (B, T, D) -> (B, D, T)
        timesteps_embed = self.diffusion_step_encoder(timestep)
        global_feature = torch.cat([timesteps_embed, global_cond], dim=-1)
        encoder_skip_features = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        x = x.permute(0, 2, 1)  # (B, D, T) -> (B, T, D)
        return x

class DiffusionPolicy(nn.Module):
    def __init__(self, policy_config):
        super().__init__()
        self.policy_config = policy_config
        self.chunk_size = policy_config['chunk_size']
        self.camera_names = policy_config['camera_names']
        self.state_dim = policy_config['state_dim']
        self.action_dim = policy_config['action_dim']

        # Backbone for image encoding
        backbone_model = torchvision.models.resnet18(pretrained=True)
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={'layer4': 'feature_map'})
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)

        # Compute global conditioning dimension
        num_cameras = len(self.camera_names)
        feat_dim = 512  # ResNet18 layer4 output channels
        global_cond_dim = self.state_dim + num_cameras * feat_dim

        # Diffusion UNet
        self.unet = DiffusionUnet(action_dim=self.action_dim, global_cond_dim=global_cond_dim)

        # Noise scheduler
        self.scheduler = DDPMScheduler(num_train_timesteps=100)

    def __call__(self, qpos, image, actions=None, is_pad=None):
        B = qpos.size(0)
        # Encode observation
        image_flat = image.view(-1, *image.shape[2:])  # (B*num_cameras, C, H, W)
        feat_map = self.backbone(image_flat)['feature_map']
        feat = self.pool(feat_map)
        feat = self.flatten(feat)
        feat = feat.view(B, -1)
        global_cond = torch.cat([qpos, feat], dim=1)

        if actions is not None:  # Training
            trajectory = actions[:, :self.chunk_size]  # Slice to chunk_size
            timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=trajectory.device)
            noise = torch.randn_like(trajectory)
            noisy_trajectory = self.scheduler.add_noise(trajectory, noise, timesteps)
            pred = self.unet(noisy_trajectory, timesteps, global_cond)
            loss = F.mse_loss(pred, noise)
            return {'loss': loss}
        else:  # Inference
            sample = torch.randn((B, self.chunk_size, self.action_dim), device=global_cond.device)
            for t in range(self.scheduler.num_train_timesteps - 1, -1, -1):
                timesteps = torch.full((B,), t, device=global_cond.device)
                model_output = self.unet(sample, timesteps, global_cond)
                sample = self.scheduler.step(model_output, t, sample).prev_sample
            return sample

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.policy_config['lr'])
