import time
from dataclasses import dataclass
from typing import Literal
import math
import time
from dataclasses import dataclass
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

ResampleMode = Literal["none", "upsample", "downsample"]


def main():
    config = Config()
    model = UNet(config)
    batch = mx.zeros((32, 64, 64, 3))
    optimizer = optim.Adam(learning_rate=0.001)

    state = [model.state, optimizer.state]

    def compute_loss(model, inp):
        output = model(inp)
        loss = mx.mean(output**2)
        return loss

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inp):
        loss_fn = nn.value_and_grad(model, compute_loss)
        loss, grads = loss_fn(model, inp)
        optimizer.update(model, grads)
        return loss

    print("warming up...")
    for _ in range(5):
        step(batch)
    mlx.core.eval(model.parameters())

    print("benchmarking...")
    t1 = time.time()
    num_its = 20
    for _ in range(num_its):
        step(batch)
    mlx.core.eval(model.parameters())

    elapsed = (time.time() - t1) / num_its
    est_flops = 684233588736.0

    gflops = est_flops / (elapsed * 1e9)
    print(f"gflops: {gflops}")


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        B, H, W, C = x.shape
        x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, 2, W, 2, C))
        return x.reshape(B, H * 2, W * 2, C)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        resample: ResampleMode = "none",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.resample = resample

        if resample == "upsample":
            self.upsample = Upsample(channels=in_channels)
            self.upsample_skip = Upsample(channels=in_channels)
        elif resample == "downsample":
            self.downsample = Downsample(channels=in_channels)
            self.downsample_skip = Downsample(channels=in_channels)
        else:
            self.upsample = self.upsample_skip = None
            self.downsample = self.downsample_skip = None

        if self.out_channels != self.in_channels:
            self.skip_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=3,
                padding=1,
            )
        else:
            self.skip_conv = None

        self.input_norm = nn.GroupNorm(num_groups=32, dims=self.in_channels)
        self.input_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
        )
        self.output_norm = nn.GroupNorm(num_groups=32, dims=self.out_channels)
        self.output_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.input_norm(x)
        if self.resample == "upsample":
            h = self.upsample(h)
            x = self.upsample_skip(x)
        elif self.resample == "downsample":
            h = self.downsample(h)
            x = self.downsample_skip(x)

        h = mlx.nn.silu(h)
        h = self.input_conv(h)

        h = self.output_norm(h)
        h = mlx.nn.silu(h)
        h = self.output_conv(h)

        if self.skip_conv:
            x = self.skip_conv(x)

        assert x.shape == h.shape, f"{x.shape} must be equal to {h.shape}"
        return x + h


@dataclass
class Config:
    in_channels: int = 3
    out_channels: int = 3
    res_block_count: int = 2
    inner_channels: list[int] = (32, 64, 64, 128)


class OutputBlock(nn.Module):
    def __init__(self, input_block: nn.Module, upsample_block: nn.Module = None):
        super().__init__()
        self.input_block = input_block
        self.upsample_block = upsample_block
        self.has_upsample = upsample_block is not None

    def __call__(self, x: mx.array) -> mx.array:
        h = self.input_block(x)
        if self.has_upsample:
            h = self.upsample_block(h)
        return h


class UNet(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.input_conv = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.inner_channels[0],
            kernel_size=3,
            padding=1,
        )

        skip_channels = [config.inner_channels[0]]
        ch = config.inner_channels[0]

        # Input Blocks
        input_blocks = []
        for i in range(1, len(config.inner_channels)):
            new_ch = config.inner_channels[i]
            for _ in range(config.res_block_count):
                input_blocks.append(ResBlock(in_channels=ch, out_channels=new_ch))
                ch = new_ch
                skip_channels.append(ch)

            if i + 1 < len(config.inner_channels):
                input_blocks.append(
                    ResBlock(in_channels=ch, out_channels=new_ch, resample="downsample")
                )
                skip_channels.append(ch)
        self.input_blocks = nn.Sequential(*input_blocks)

        # Middle Blocks
        middle_blocks = [ResBlock(in_channels=ch), ResBlock(in_channels=ch)]
        self.middle_blocks = nn.Sequential(*middle_blocks)

        # Output Blocks
        output_blocks = []
        for i in range(len(config.inner_channels) - 1, 0, -1):
            out_channels = config.inner_channels[i - 1]
            for j in range(config.res_block_count + 1):
                skip = skip_channels.pop()
                input_block = ResBlock(in_channels=ch + skip, out_channels=out_channels)
                ch = out_channels
                upsample = (
                    ResBlock(in_channels=ch, resample="upsample")
                    if i > 1 and j == config.res_block_count
                    else None
                )
                output_blocks.append(OutputBlock(input_block, upsample))
        self.output_blocks = nn.Sequential(*output_blocks)

        self.output_norm = nn.GroupNorm(num_groups=32, dims=config.inner_channels[0])
        self.output_conv = nn.Conv2d(
            in_channels=config.inner_channels[0],
            out_channels=config.out_channels,
            kernel_size=3,
            padding=1,
        )

    def __call__(self, x: mx.array) -> mx.array:
        h = self.input_conv(x)
        skips = [h]

        # Process input blocks
        for block in self.input_blocks.layers:
            h = block(h)
            skips.append(h)

        # Process middle blocks
        h = self.middle_blocks(h)

        # Process output blocks
        for block, skip in zip(self.output_blocks.layers, reversed(skips)):
            h = mx.concatenate([h, skip], axis=-1)
            h = block(h)

        h = self.output_norm(h)
        h = mlx.nn.silu(h)
        h = self.output_conv(h)
        return h


if __name__ == "__main__":
    main()
