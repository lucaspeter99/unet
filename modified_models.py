import torch
from torch import nn
from collections import OrderedDict


# TODO: dinamically change network with in_channels
class UNet(nn.Module): # Max Pooling, BatchNorm & LeakyReLU on demand
    """
    The layout of the model has been adapted from https://github.com/mateuszbuda/brain-segmentation-pytorch,
    using 3-dimensional kernels and other modifications.


    MIT License

Copyright (c) 2019 mateuszbuda

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""
    def __init__(self, in_channels, out_channels, features, batchnorm=True, leaky=True, max_pool=True):
        '''
        :param in_channels: modalities
        :param out_channels: output modalitites
        :param features: size of input (each modality)
        :param batchnorm: bool
        :param leaky: bool
        :param max_pool: bool
        '''
        super(UNet, self).__init__()
        self.batchnorm = batchnorm
        self.leaky = leaky
        self.in_channels = in_channels

        self.encoder11 = self._block(1, features, name="enc11")
        self.pool11 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder21 = self._block(features, features * 2, name="enc21")
        self.pool21 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder31 = self._block(features * 2, features * 4, name="enc31")
        self.pool31 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        self.encoder41 = self._block(features * 4, features * 8, name="enc41")
        self.pool41 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        if in_channels > 1 :
            self.encoder12 = self._block(1, features, name="enc12")
            self.pool12 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder22 = self._block(features, features * 2, name="enc22")
            self.pool22 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder32 = self._block(features * 2, features * 4, name="enc32")
            self.pool32 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder42 = self._block(features * 4, features * 8, name="enc42")
            self.pool42 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if in_channels > 2:
            self.encoder13 = self._block(1, features, name="enc13")
            self.pool13 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder23 = self._block(features, features * 2, name="enc23")
            self.pool23 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder33 = self._block(features * 2, features * 4, name="enc33")
            self.pool33 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder43 = self._block(features * 4, features * 8, name="enc43")
            self.pool43 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if in_channels > 3:
            self.encoder14 = self._block(1, features, name="enc14")
            self.pool14 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder24 = self._block(features, features * 2, name="enc24")
            self.pool24 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder34 = self._block(features * 2, features * 4, name="enc34")
            self.pool34 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder44 = self._block(features * 4, features * 8, name="enc44")
            self.pool44 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
        if in_channels > 4:
            self.encoder15 = self._block(1, features, name="enc15")
            self.pool15 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder25 = self._block(features, features * 2, name="enc25")
            self.pool25 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder35 = self._block(features * 2, features * 4, name="enc35")
            self.pool35 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)
            self.encoder45 = self._block(features * 4, features * 8, name="enc45")
            self.pool45 = nn.MaxPool3d(kernel_size=2, stride=2) if max_pool else nn.AvgPool3d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8 * in_channels, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose3d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * (self.in_channels+1), features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * (self.in_channels+1), features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * (self.in_channels+1), features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * (self.in_channels+1), features, name="dec1")

        self.conv = nn.Conv3d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):

        mod_list = []

        for batch in x:
            #[b1, b2, b3, b4, b5] = batch

            if len(mod_list) is 0:
                for j in range(self.in_channels):
                    xj = torch.unsqueeze(batch[j], 0)
                    mod_list.append(xj)
            else:
                for j in range(self.in_channels):
                    xj = torch.unsqueeze(batch[j], 0)
                    item = mod_list[j]
                    mod_list[j] = torch.cat((item, xj), 0)

                # x1 = torch.cat((x1, torch.unsqueeze(b1, 0)), 0)
                # x2 = torch.cat((x2, torch.unsqueeze(b2, 0)), 0)
                # x3 = torch.cat((x3, torch.unsqueeze(b3, 0)), 0)
                # x4 = torch.cat((x4, torch.unsqueeze(b4, 0)), 0)
                # x5 = torch.cat((x5, torch.unsqueeze(b5, 0)), 0)

        # [x1, x2, x3, x4, x5] = x[0]

        enc11 = self.encoder11(torch.unsqueeze(mod_list[0],1))
        enc21 = self.encoder21(self.pool11(enc11))
        enc31 = self.encoder31(self.pool21(enc21))
        enc41 = self.encoder41(self.pool31(enc31))
        pool41 = self.pool41(enc41)

        if self.in_channels > 1:
            enc12 = self.encoder12(torch.unsqueeze(mod_list[1],1))
            enc22 = self.encoder22(self.pool12(enc12))
            enc32 = self.encoder32(self.pool22(enc22))
            enc42 = self.encoder42(self.pool32(enc32))
            pool42 = self.pool41(enc42)
        if self.in_channels > 2:
            enc13 = self.encoder13(torch.unsqueeze(mod_list[2],1))
            enc23 = self.encoder23(self.pool13(enc13))
            enc33 = self.encoder33(self.pool23(enc23))
            enc43 = self.encoder43(self.pool33(enc33))
            pool43 = self.pool41(enc43)
        if self.in_channels > 3:
            enc14 = self.encoder14(torch.unsqueeze(mod_list[3],1))
            enc24 = self.encoder24(self.pool14(enc14))
            enc34 = self.encoder34(self.pool24(enc24))
            enc44 = self.encoder44(self.pool34(enc34))
            pool44 = self.pool41(enc44)
        if self.in_channels > 4:
            enc15 = self.encoder15(torch.unsqueeze(mod_list[4],1))
            enc25 = self.encoder25(self.pool15(enc15))
            enc35 = self.encoder35(self.pool25(enc25))
            enc45 = self.encoder45(self.pool35(enc35))
            pool45 = self.pool41(enc45)

        if self.in_channels > 4:
            bottle_input = torch.cat((pool41, pool42, pool43, pool44, pool45), dim = 1)
            dec4_input = torch.cat((enc41, enc42, enc43, enc44, enc45), dim=1)
            dec3_input = torch.cat((enc31, enc32, enc33, enc34, enc35), dim=1)
            dec2_input = torch.cat((enc21, enc22, enc23, enc24, enc25), dim=1)
            dec1_input = torch.cat((enc11, enc12, enc13, enc14, enc15), dim=1)
        elif self.in_channels > 3:
            bottle_input = torch.cat((pool41, pool42, pool43, pool44), dim=1)
            dec4_input = torch.cat((enc41, enc42, enc43, enc44), dim=1)
            dec3_input = torch.cat((enc31, enc32, enc33, enc34), dim=1)
            dec2_input = torch.cat((enc21, enc22, enc23, enc24), dim=1)
            dec1_input = torch.cat((enc11, enc12, enc13, enc14), dim=1)

        elif self.in_channels > 2:
            bottle_input = torch.cat((pool41, pool42, pool43), dim=1)
            dec4_input = torch.cat((enc41, enc42, enc43), dim=1)
            dec3_input = torch.cat((enc31, enc32, enc33), dim=1)
            dec2_input = torch.cat((enc21, enc22, enc23), dim=1)
            dec1_input = torch.cat((enc11, enc12, enc13), dim=1)
        elif self.in_channels > 1:
            bottle_input = torch.cat((pool41, pool42), dim=1)
            dec4_input = torch.cat((enc41, enc42), dim=1)
            dec3_input = torch.cat((enc31, enc32), dim=1)
            dec2_input = torch.cat((enc21, enc22), dim=1)
            dec1_input = torch.cat((enc11, enc12), dim=1)


        bottleneck = self.bottleneck(bottle_input)

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, dec4_input), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, dec3_input), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, dec2_input), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, dec1_input), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    def _block(self, in_channels, features, name):
        if self.batchnorm:
            activation = nn.Sequential(
                nn.BatchNorm3d(num_features=features),
                nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )
        else:
            activation = nn.Sequential(
                nn.LeakyReLU(inplace=True) if self.leaky else nn.ReLU(inplace=True)
            )

        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "activation1", activation),
                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "activation2", activation),
                ]
            )
        )


class Discriminator3d(nn.Module):

    def __init__(self, in_channels, out_channels, features, batchnorm=True):
        super(Discriminator3d, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(features * 4, features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(features * 8),
            nn.Conv3d(features * 8, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        ) if batchnorm else nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(features * 4, features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Conv3d(features * 8, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        return self.main(input_tensor)[0, 0, 0, 0].cuda()



