import torch
import torch.nn as nn

# -------- Basic Residual Block (no BN), from EDSR paper --------
class ResidualBlock(nn.Module):
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        return x + res


# -------- Upsampler (PixelShuffle) --------
class Upsampler(nn.Module):
    def __init__(self, scale, n_feats):
        super().__init__()
        m = []
        if scale in (2, 4):
            n_up = 2 if scale == 4 else 1
            for _ in range(n_up):
                m += [
                    nn.Conv2d(n_feats, 4 * n_feats, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(2),
                ]
        elif scale == 3:
            m += [
                nn.Conv2d(n_feats, 9 * n_feats, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(3),
            ]
        else:
            raise ValueError(f"Unsupported scale {scale}. Use 2, 3, or 4.")
        self.upsample = nn.Sequential(*m)

    def forward(self, x):
        return self.upsample(x)


# -------- EDSR Model --------
class EDSR(nn.Module):
    def __init__(self, scale=2, n_resblocks=16, n_feats=64, n_colors=1, res_scale=0.1):
        super().__init__()
        self.scale = scale

        # Head
        self.head = nn.Conv2d(n_colors, n_feats, kernel_size=3, stride=1, padding=1)

        # Body
        body = [ResidualBlock(n_feats, res_scale=res_scale) for _ in range(n_resblocks)]
        body += [nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)]
        self.body = nn.Sequential(*body)

        # Upsampler + Tail
        self.upsampler = Upsampler(scale, n_feats)
        self.tail = nn.Conv2d(n_feats, n_colors, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.upsampler(res)
        x = self.tail(x)
        return x


# -------- Model factory function --------
def edsr_x(scale=2, n_resblocks=16, n_feats=64, n_colors=1, res_scale=0.1):
    """Factory function to build EDSR model."""
    return EDSR(scale, n_resblocks, n_feats, n_colors, res_scale)
