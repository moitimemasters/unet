import torch
import torch.nn as nn
import torchvision.transforms.functional as F


DEFAULT_FEATURES = [32, 64, 128, 256, 512]

class Conv2(nn.Module):
    def __init__(self, in_, out):
        super(Conv2, self).__init__()
        self.in_ = in_
        self.out = out
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_, self.out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.out),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out, self.out, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Unet(nn.Module):
    def __init__(self, in_, out, features):
        super(Unet, self).__init__()
        self.in_ = in_
        self.out = out
        self.features = features
        self.first_half = nn.ModuleList()
        self.second_half = nn.ModuleList()
        self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bridge = Conv2(features[-1], features[-1] * 2)
        self.set_first_half()
        self.set_second_half()
        self.output_layer = nn.Conv2d(features[0], self.out, kernel_size=1)

    def set_first_half(self):
        for feature in self.features:
            self.first_half.append(Conv2(self.in_, feature))
            self.in_ = feature

    def set_second_half(self):
        for feature in self.features[::-1]:
            self.second_half.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.second_half.append(Conv2(feature * 2, feature))            
                    

    def forward(self, x):
        to_connect = []
        for layer in self.first_half:
            x = layer(x)
            to_connect.append(x)
            x = self.pool_layer(x) 
        x = self.bridge(x)
        pair_second = zip(*[iter(self.second_half)]*2)
        for (layer, concat_layer), connection in zip(pair_second, reversed(to_connect)):
            x = layer(x)
            if x.shape != connection.shape:
                x = F.resize(x, size=connection.shape[2:])
            concatenated = torch.cat((connection, x), dim=1)
            x = concat_layer(concatenated)

        return self.output_layer(x)

if __name__ == "__main__":
    x = torch.randn((3, 1, 161, 161))
    model = Unet(in_=1, out=1, features=DEFAULT_FEATURES)
    preds = model(x)
    print(x.shape)
    assert preds.shape == x.shape
    
        
       
