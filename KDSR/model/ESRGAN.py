from model.block import *


class ESRGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nf=64, gc=32, scale_factor=4, n_basic_block=23):
        super(ESRGAN, self).__init__()

        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(in_channels, nf, 3), nn.ReLU())

        basic_block_layer = []

        for _ in range(n_basic_block):
            basic_block_layer += [ResidualInResidualDenseBlock(nf, gc)]

        self.basic_block = nn.ModuleList( basic_block_layer)

        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, 3), nn.ReLU())
        self.upsample = upsample_block(nf, scale_factor=scale_factor)
        self.conv3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, nf, 3), nn.ReLU())
        self.conv4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(nf, out_channels, 3), nn.ReLU())

    def forward(self, x):
        feature_maps = []
        x1 = self.conv1(x)
        for group_id,layer in enumerate(self.basic_block):
            x = layer(x1)
            if group_id == 2 or group_id == 6:
                feature_maps.append(x)
        x = self.conv2(x)
        feature_maps.append(x)
        x = self.upsample(x + x1)
        x = self.conv3(x)
        x = self.conv4(x)
        return feature_maps,x
