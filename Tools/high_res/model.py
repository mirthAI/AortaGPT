import torch.nn as nn
import torch
import torch.nn.functional as F


# -------------------------------
# Embedded RDN encoder (3D)
# -------------------------------
class _DenseLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.relu(self.conv(x))], dim=1)


class _ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, num_layers: int):
        super().__init__()
        self.layers = nn.Sequential(*[
            _DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)
        ])
        # Match original key name 'lff' for compatibility with saved weights
        self.lff = nn.Conv3d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lff(self.layers(x))


class RDN(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        num_features: int = 64,
        growth_rate: int = 64,
        num_blocks: int = 8,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        G0 = num_features
        G = growth_rate
        D = num_blocks
        C = num_layers

        self.sfe1 = nn.Conv3d(1, G0, kernel_size=3, padding=1)
        self.sfe2 = nn.Conv3d(G0, G0, kernel_size=3, padding=1)

        self.rdbs = nn.ModuleList([_ResidualDenseBlock(G0, G, C)])
        for _ in range(D - 1):
            self.rdbs.append(_ResidualDenseBlock(G, G, C))

        # Match original key name 'gff' for compatibility with saved weights
        self.gff = nn.Sequential(
            nn.Conv3d(G * D, G0, kernel_size=1),
            nn.Conv3d(G0, G0, kernel_size=3, padding=1),
        )
        self.output = nn.Conv3d(G0, feature_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sfe1 = self.sfe1(x)
        x = self.sfe2(sfe1)
        local_features = []
        for rdb in self.rdbs:
            x = rdb(x)
            local_features.append(x)
        x = self.gff(torch.cat(local_features, dim=1)) + sfe1
        return self.output(x)


# -------------------------------
# Embedded MLP decoder
# -------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int = 128 + 3, out_dim: int = 1, depth: int = 4, width: int = 256) -> None:
        super().__init__()
        stage_one_layers = []
        stage_two_layers = []
        for i in range(depth):
            if i == 0:
                stage_one_layers.append(nn.Linear(in_dim, width))
                stage_two_layers.append(nn.Linear(in_dim, width))
            elif i == depth - 1:
                stage_one_layers.append(nn.Linear(width, in_dim))
                stage_two_layers.append(nn.Linear(width, out_dim))
            else:
                stage_one_layers.append(nn.Linear(width, width))
                stage_two_layers.append(nn.Linear(width, width))
            stage_one_layers.append(nn.ReLU())
            stage_two_layers.append(nn.ReLU())
        self.stage_one = nn.Sequential(*stage_one_layers)
        self.stage_two = nn.Sequential(*stage_two_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stage_one(x)
        return self.stage_two(x + h)


# -------------------------------
# ArSSR model (portable, RDN-only)
# -------------------------------
class ArSSR(nn.Module):
    def __init__(self, encoder_name: str = "RDN", feature_dim: int = 128, decoder_depth: int = 4, decoder_width: int = 256):
        super().__init__()
        if encoder_name != "RDN":
            raise ValueError("model_portable.ArSSR only supports encoder_name='RDN'")
        self.encoder = RDN(feature_dim=feature_dim)
        self.decoder = MLP(in_dim=feature_dim + 3, out_dim=1, depth=decoder_depth, width=decoder_width)

    def forward(self, img_lr: torch.Tensor, dhw_hr: torch.Tensor) -> torch.Tensor:
        # extract feature map from LR image
        feature_map = self.encoder(img_lr)  # N×C×d×h×w where C=feature_dim
        # grid_sample expects (x, y, z) but dhw_hr is (z, y, x); flip last dim
        feature_vector = (
            F.grid_sample(
                feature_map,
                dhw_hr.flip(-1).unsqueeze(1).unsqueeze(1),
                mode="bilinear",
                align_corners=False,
            )
            [:, :, 0, 0, :]
            .permute(0, 2, 1)
        )
        feature_vector_and_dhw_hr = torch.cat([feature_vector, dhw_hr], dim=-1)  # N×K×(3+feature_dim)
        N, K = dhw_hr.shape[:2]
        intensity_pre = self.decoder(feature_vector_and_dhw_hr.view(N * K, -1)).view(N, K, -1)
        return intensity_pre
 