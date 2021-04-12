import torch
import torch.nn as nn

# no batch normalization layer because the batch size is small in our experiment
# if batch normalization layers are added, Conv2d layers no longer need bias
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, negative_slope=0.1):
        super(ConvBlock, self).__init__()
        
        self.ConvLayer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.BNLayer = nn.BatchNorm2d(out_channels)
        self.ReluLayer = nn.LeakyReLU(negative_slope, inplace=True)
        
    def forward(self, input):
        output = self.ConvLayer(input)
        output = self.ReluLayer(output)
        return output
    
# using architecture of YoloV2
# assume image_size=448
# assume split_size=7
class YoloV2mod(nn.Module):
    def __init__(self, num_boxes=2, num_classes=2):
        super(YoloV2mod, self).__init__()
        
        self.darknet19_half1 = nn.Sequential(
                ConvBlock(3, 32, 3, 1, 1),
                nn.MaxPool2d(2, 2),
                ConvBlock(32, 64, 3, 1, 1),
                nn.MaxPool2d(2, 2),
                ConvBlock(64, 128, 3, 1, 1),
                ConvBlock(128, 64, 1, 1, 0),
                ConvBlock(64, 128, 3, 1, 1),
                nn.MaxPool2d(2, 2),
                ConvBlock(128, 256, 3, 1, 1),
                ConvBlock(256, 128, 1, 1, 0),
                ConvBlock(128, 256, 3, 1, 1),
                nn.MaxPool2d(2, 2),
                ConvBlock(256, 512, 3, 1, 1),
                ConvBlock(512, 256, 1, 1, 0),
                ConvBlock(256, 512, 3, 1, 1),
                ConvBlock(512, 256, 1, 1, 0),
                ConvBlock(256, 512, 3, 1, 1),
                nn.MaxPool2d(2, 2)
                )

        self.passThroughPre = ConvBlock(512, 64, 1, 1, 0)

        # modified
        self.darknet19_half2 = nn.Sequential(
                nn.MaxPool2d(2, 2),
                ConvBlock(512, 1024, 3, 1, 1),
                ConvBlock(1024, 512, 1, 1, 0),
                ConvBlock(512, 1024, 3, 1, 1),
                ConvBlock(1024, 512, 1, 1, 0),
                ConvBlock(512, 1024, 3, 1, 1)
                )
        
        self.preConcat = nn.Sequential(
                ConvBlock(1024, 1024, 3, 1, 1),
                ConvBlock(1024, 1024, 3, 1, 1)
                )
        
        self.postConcat = nn.Sequential(
                ConvBlock(1024+256, 1024, 3, 1, 1),
                nn.Conv2d(1024, 5*num_boxes+num_classes, 1, 1, 0, bias=False),
                nn.Flatten()
                )

    def forward(self, input):
        output_half1 = self.darknet19_half1(input)
        
        passThrough = self.passThroughPre(output_half1)
        batch_size, num_channel, height, width = passThrough.data.size()
        passThrough = passThrough.view(batch_size, int(num_channel / 4), height, 2, width, 2).contiguous()
        passThrough = passThrough.permute(0, 3, 5, 1, 2, 4).contiguous()
        passThrough = passThrough.view(batch_size, -1, int(height / 2), int(width / 2))

        output_half2 = self.darknet19_half2(output_half1)
        output_main = self.preConcat(output_half2)

        output = torch.cat((output_main, passThrough), 1)
        output = self.postConcat(output)
        return output

def test(S=7, B=2, C=2):
    model = YoloV2mod(num_boxes=B, num_classes=C)
    input = torch.randn((2, 3, 448, 448))
    print(model(input).shape)

if __name__ == "__main__":
    test()
    