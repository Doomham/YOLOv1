import torch.nn as nn
import torch
from backbone.resnet import resnet101
import torch.nn.functional as F
from backbone.darknet import DarkNet, darknet19


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class YOLOv1(nn.Module):
    def __init__(self, features, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C

        self.backbone = features
        self.conv_layers = nn.Sequential(
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
                nn.LeakyReLU(0.1),

                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(1024, 1024, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True))
        self.fc_layers = nn.Sequential(
            Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Linear(4096, S * S * (5 * B + C)),
            nn.Sigmoid())

    def forward(self, x):
        f = self.backbone(x)
        f = self.conv_layers(f)
        f = self.fc_layers(f)
        f = f.view(-1, self.S, self.S, 5 * self.B + self.C)
        return f


def test():
    from torch.autograd import Variable

    # Build model with randomly initialized weights
    darknet = DarkNet(conv_only=True, bn=True, init_weight=True)
    #yolo = YOLOv1(darknet.features, 7, 2, 20)
    #yolo = YOLOv1(resnet101())
    # Prepare a dummy image to input
    image = torch.rand(2, 3, 448, 448)
    image = Variable(image)

    # Forward
    #output = yolo(image)
    feature = darknet.features(image)
    print(feature.shape)
    new_darknet = darknet19()
    f = new_darknet(image)
    print(f.shape)
    # Check ouput tensor size, which should be [2, 7, 7, 30]
    #print(output.size())


if __name__ == '__main__':
    test()