import torch

class ConvBlock(torch.nn.Module):
    """
    下采样和上采样的相同结构
    """
    def __init__(self, in_channels, out_channels, padding):
        super(ConvBlock, self).__init__()
        blocks = []
        blocks.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding))
        blocks.append(torch.nn.BatchNorm2d(out_channels))
        blocks.append(torch.nn.ReLU())

        blocks.append(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding))
        blocks.append(torch.nn.BatchNorm2d(out_channels))
        blocks.append(torch.nn.ReLU())
        
        self.blocks = torch.nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels, padding, pool=None):
        super(Down, self).__init__()

        self.maxpool = None
        if pool is not None:
            self.maxpool = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_block = ConvBlock(in_channels, out_channels, padding)

    def forward(self, x):
        if self.maxpool is not None:
            x = self.maxpool(x)
        return self.conv_block(x)

class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels, padding):
        super(Up, self).__init__()

        self.up = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_channels, out_channels, padding)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        assert layer_height >= target_size[0]
        assert layer_width >= target_size[1]
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print(x1.shape, x2.shape)
        x2 = self.center_crop(x2, x1.shape[2:])
        # print(x1.shape, x2.shape)
        x = torch.cat([x1, x2], 1)
        # print(x.shape)
        x = self.conv_block(x)
        return x

class Unet(torch.nn.Module):
    def __init__(self, in_channels, n_classes):
        super(Unet, self).__init__()

        self.down1 = Down(in_channels, 64, 0, None)
        self.down2 = Down(64, 128, 0, 'MaxPool')
        self.down3 = Down(128, 256, 0, 'MaxPool')
        self.down4 = Down(256, 512, 0, 'MaxPool')
        self.down5 = Down(512, 1024, 0, 'MaxPool')

        self.up1 = Up(1024, 512, 0)
        self.up2 = Up(512, 256, 0)
        self.up3 = Up(256, 128, 0)
        self.up4 = Up(128, 64, 0)

        self.last = torch.nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        # print(x1.shape)
        x2 = self.down2(x1)
        # print(x2.shape)
        x3 = self.down3(x2)
        # print(x3.shape)
        x4 = self.down4(x3)
        # print(x4.shape)
        x5 = self.down5(x4)
        # print(x5.shape)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.last(x)
        return x

if __name__ == "__main__":
    x = torch.rand((1, 3, 572, 572))

    model = Unet(3, 10)

    out = model(x)
    print(x.shape, out.shape)