import torch
import torch.nn as nn

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class LeNet(nn.Module):
    def __init__(self, input_shape):
        super(LeNet, self).__init__()

        # Encoder: 4 conv blocks (Conv2d -> LeakyReLU -> BatchNorm2d) per paper §3.2.
        # Indices below are referenced from forward() to inject residuals between blocks.
        # [0,1,2]   = conv1, lrelu, bn1
        # [3,4,5]   = conv2, lrelu, bn2
        # [6,7,8]   = conv3, lrelu, bn3
        # [9,10,11] = conv4, lrelu, bn4
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
        )

        # Decoder: mirrored transposed convs with BN per paper §3.2; final Sigmoid.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

        # Residual connections after the first and second conv layers (paper §3.2).
        self.residual1 = nn.Conv2d(3, 32, 1)
        self.residual2 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # Block 1: conv1 + lrelu, add residual, then bn1.
        res1 = self.residual1(x)
        e1 = self.encoder[0:2](x)
        e1 = e1 + res1
        e1 = self.encoder[2](e1)

        # Block 2: conv2 + lrelu, add residual, then bn2.
        res2 = self.residual2(e1)
        e2 = self.encoder[3:5](e1)
        e2 = e2 + res2
        e2 = self.encoder[5](e2)

        # Blocks 3 and 4: remaining conv+lrelu+bn pairs.
        features = self.encoder[6:](e2)

        reconstructed = self.decoder(features)
        return reconstructed