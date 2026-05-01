# -*- coding: UTF-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=timesteps.device) * -embeddings
        )
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FiLMLayer(nn.Module):
    def __init__(self, in_channels, time_embed_dim, class_embed_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.time_proj = nn.Linear(time_embed_dim, in_channels * 2)
        self.class_proj = nn.Linear(class_embed_dim, in_channels * 2)

    def forward(self, x, time_emb=None, class_emb=None):
        h = self.norm(x)
        if time_emb is not None:
            time_scale_shift = self.time_proj(time_emb)
            scale, shift = time_scale_shift.chunk(2, dim=1)
            h = h * (1 + scale) + shift
        if class_emb is not None:
            class_scale_shift = self.class_proj(class_emb)
            scale, shift = class_scale_shift.chunk(2, dim=1)
            h = h * (1 + scale) + shift
        return h


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, time_embed_dim, class_embed_dim, dropout=0.1
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_embed_dim, out_channels * 2)
        self.class_proj = nn.Linear(class_embed_dim, out_channels * 2)
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb=None, c_emb=None):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        if t_emb is not None:
            t_scale_shift = self.time_proj(F.silu(t_emb))
            scale, shift = t_scale_shift.chunk(2, dim=1)
            h = h * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(
                -1
            ).unsqueeze(-1)

        if c_emb is not None:
            c_scale_shift = self.class_proj(F.silu(c_emb))
            scale, shift = c_scale_shift.chunk(2, dim=1)
            h = h * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(
                -1
            ).unsqueeze(-1)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_embed_dim,
        class_embed_dim,
        num_layers=2,
        downsample=True,
        dropout=0.1,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    time_embed_dim,
                    class_embed_dim,
                    dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.downsample = downsample
        if downsample:
            self.pool = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )

    def forward(self, x, t_emb=None, c_emb=None):
        for res_block in self.res_blocks:
            x = res_block(x, t_emb, c_emb)
        if self.downsample:
            x = self.pool(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels,
        time_embed_dim,
        class_embed_dim,
        num_layers=2,
        upsample=True,
        dropout=0.1,
    ):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.res_blocks.append(
                    ResidualBlock(
                        in_channels + skip_channels,
                        out_channels,
                        time_embed_dim,
                        class_embed_dim,
                        dropout,
                    )
                )
            else:
                self.res_blocks.append(
                    ResidualBlock(
                        out_channels,
                        out_channels,
                        time_embed_dim,
                        class_embed_dim,
                        dropout,
                    )
                )
        self.upsample = upsample
        if upsample:
            self.up = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=4, stride=2, padding=1
            )

    def forward(self, x, skip_x, t_emb=None, c_emb=None):
        x = torch.cat([x, skip_x], dim=1)
        for res_block in self.res_blocks:
            x = res_block(x, t_emb, c_emb)
        if self.upsample:
            x = self.up(x)
        return x


class MidBlock(nn.Module):
    def __init__(self, channels, time_embed_dim, class_embed_dim, dropout=0.1):
        super().__init__()
        self.res1 = ResidualBlock(
            channels, channels, time_embed_dim, class_embed_dim, dropout
        )
        self.attention = SelfAttention(channels)
        self.res2 = ResidualBlock(
            channels, channels, time_embed_dim, class_embed_dim, dropout
        )

    def forward(self, x, t_emb=None, c_emb=None):
        x = self.res1(x, t_emb, c_emb)
        x = self.attention(x)
        x = self.res2(x, t_emb, c_emb)
        return x


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attention(h, h, h)
        h = self.proj(h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h


class ClassConditionalUNet(nn.Module):
    def __init__(
        self,
        num_classes=100,
        in_channels=3,
        out_channels=3,
        sample_size=32,
        time_embed_dim=512,
        class_embed_dim=512,
        block_out_channels=(128, 256, 256, 512),
        layers_per_block=2,
        dropout=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.sample_size = sample_size
        self.time_embed_dim = time_embed_dim
        self.class_embed_dim = class_embed_dim

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        self.class_embedding = nn.Embedding(num_classes + 1, class_embed_dim)
        self.class_proj = nn.Sequential(
            nn.Linear(class_embed_dim, class_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(class_embed_dim * 4, class_embed_dim),
        )

        self.conv_in = nn.Conv2d(
            in_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        in_ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            is_last = i == len(block_out_channels) - 1
            self.down_blocks.append(
                DownBlock(
                    in_ch,
                    out_ch,
                    time_embed_dim,
                    class_embed_dim,
                    num_layers=layers_per_block,
                    downsample=not is_last,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.mid_block = MidBlock(
            block_out_channels[-1], time_embed_dim, class_embed_dim, dropout=dropout
        )

        reversed_channels = list(reversed(block_out_channels))
        for i in range(len(block_out_channels)):
            skip_ch = (
                block_out_channels[-(i + 2)]
                if i + 2 <= len(block_out_channels)
                else block_out_channels[0]
            )
            in_ch = reversed_channels[i]
            out_ch = (
                reversed_channels[i + 1]
                if i + 1 < len(reversed_channels)
                else reversed_channels[i]
            )

            self.up_blocks.append(
                UpBlock(
                    in_ch,
                    out_ch,
                    skip_ch,
                    time_embed_dim,
                    class_embed_dim,
                    num_layers=layers_per_block,
                    upsample=i < len(block_out_channels) - 1,
                    dropout=dropout,
                )
            )

        self.conv_norm_out = nn.GroupNorm(
            num_groups=32, num_channels=block_out_channels[0], eps=1e-6
        )
        self.conv_out = nn.Conv2d(
            block_out_channels[0], out_channels, kernel_size=3, padding=1
        )

    def forward(self, sample, timestep, class_labels=None):
        t_emb = self.time_embedding(timestep)

        if class_labels is not None:
            c_emb = self.class_embedding(class_labels)
            c_emb = self.class_proj(c_emb)
        else:
            c_emb = None

        x = self.conv_in(sample)

        skip_connections = []
        for down_block in self.down_blocks:
            skip_connections.append(x)
            x = down_block(x, t_emb, c_emb)

        x = self.mid_block(x, t_emb, c_emb)

        for up_block in self.up_blocks:
            skip_x = skip_connections.pop()
            x = up_block(x, skip_x, t_emb, c_emb)

        x = self.conv_norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x


def load_pretrained_unet(
    pretrained_model_name="google/ddpm-cifar10-32", num_classes=101, device="cpu"
):
    try:
        from diffusers import UNet2DModel

        pretrained_unet = UNet2DModel.from_pretrained(pretrained_model_name)
        pretrained_config = pretrained_unet.config
        block_out_channels = tuple(pretrained_config.block_out_channels)
        layers_per_block = pretrained_config.layers_per_block
    except Exception:
        block_out_channels = (128, 256, 256, 512)
        layers_per_block = 2

    model = ClassConditionalUNet(
        num_classes=num_classes - 1,
        in_channels=3,
        out_channels=3,
        sample_size=32,
        time_embed_dim=512,
        class_embed_dim=512,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
    )

    try:
        pretrained_state = pretrained_unet.state_dict()
        new_state = model.state_dict()

        key_mapping = {
            "down_blocks.{}.resnets.{}.norm1": "down_blocks.{}.res_blocks.{}.norm1",
            "down_blocks.{}.resnets.{}.conv1": "down_blocks.{}.res_blocks.{}.conv1",
            "down_blocks.{}.resnets.{}.norm2": "down_blocks.{}.res_blocks.{}.norm2",
            "down_blocks.{}.resnets.{}.conv2": "down_blocks.{}.res_blocks.{}.conv2",
            "up_blocks.{}.resnets.{}.norm1": "up_blocks.{}.res_blocks.{}.norm1",
            "up_blocks.{}.resnets.{}.conv1": "up_blocks.{}.res_blocks.{}.conv1",
            "up_blocks.{}.resnets.{}.norm2": "up_blocks.{}.res_blocks.{}.norm2",
            "up_blocks.{}.resnets.{}.conv2": "up_blocks.{}.res_blocks.{}.conv2",
        }

        loaded_count = 0
        for new_key in new_state:
            pretrained_key = new_key
            for old_pattern, new_pattern in key_mapping.items():
                if new_pattern.replace("{}", "[0-9]+") in new_key:
                    pretrained_key = new_key
                    for i in range(10):
                        pretrained_key = pretrained_key.replace(
                            new_pattern.format(i, "{}").replace("{}", ""),
                            old_pattern.format(i, "").replace("{}", ""),
                        )
                    break

            if pretrained_key in pretrained_state:
                if pretrained_state[pretrained_key].shape == new_state[new_key].shape:
                    new_state[new_key] = pretrained_state[pretrained_key]
                    loaded_count += 1

        if "time_embedding.0.weight" in pretrained_state:
            new_state["time_embedding.0.weight"] = pretrained_state[
                "time_embedding.0.weight"
            ]
            new_state["time_embedding.0.bias"] = pretrained_state[
                "time_embedding.0.bias"
            ]
            new_state["time_embedding.2.weight"] = pretrained_state[
                "time_embedding.2.weight"
            ]
            new_state["time_embedding.2.bias"] = pretrained_state[
                "time_embedding.2.bias"
            ]

        print(f"Loaded {loaded_count} pretrained weights")

    except Exception as e:
        print(f"Could not load pretrained weights: {e}")

    model.load_state_dict(new_state)
    model.to(device)

    return model
