import math
import torch
import numpy as np
import torch.nn as nn



class VGG_part_one(nn.Module):
    
    def __init__(self, batch_norm, border, feature_dim, channels, model_size='A'):
        super(VGG_part_one, self).__init__()
        '''
        VGG model 
        '''
        self.sizes = {
        'VGG11': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
              512, 512, 512, 512, 'M'],
        }
        self.feature_dim=feature_dim
        self.transform=['D', 4096, 'R', 'D', 2048, 4096]


        self.features = self.make_layers(model_size, border, channels, batch_norm)
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
    
        return x

    def make_layers(self, model_size, border, in_channels, batch_norm=False):
        layers = []
        i=0
        for v in self.sizes[model_size]:
            if i>border:
                break
            print(v)
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
            i=i+1
        if i<=border:
            layers+=[nn.Flatten()]
        in_dim=self.feature_dim
        for v in self.transform:
            if i>border:
                break
            print(v)
            if v =='D':
                layers+=[nn.Dropout()]
            elif v =='R':
                layers+=[nn.ReLU(True)]
            else:
                layers+=[nn.Linear(in_dim, v)]
                in_dim=v
            i=i+1
        print('_____')
        return nn.Sequential(*layers)
class VGG_part_two(nn.Module):
    
    def __init__(self, batch_norm, border, out_size, feature_dim, model_size='A'):
        super(VGG_part_two, self).__init__()
        self.sizes = {
        'VGG11': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
              512, 512, 512, 512, 'M'],
        }
        self.feature_dim=feature_dim
        self.transform=['D', 2048, 'R', 'D', 2048, 'R', out_size]
        self.features=self.make_layers(model_size, border, batch_norm)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
    def make_layers(self, model_size, border, batch_norm=False):
        layers = []
        if border<len(self.sizes[model_size]):
            
            in_channels = self.sizes[model_size][border-1] if type(self.sizes[model_size][border-1])==int else self.sizes[model_size][border-2]
            i=-1
            for v in self.sizes[model_size]:
                i=i+1
                if i<=border:
                    continue
                print(v)
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v
            layers+=[nn.Flatten()]
        in_dim=self.feature_dim
        i=len(self.sizes[model_size])-1
        for v in self.transform:
            i=i+1
            if i<=border:
                if type(v)==int:
                    in_dim=v
                continue
            print(v)
            if v =='D':
                layers+=[nn.Dropout()]
            elif v =='R':
                layers+=[nn.ReLU(True)]
            else:
                layers+=[nn.Linear(in_dim, v)]
                in_dim=v
            i=i+1

        return nn.Sequential(*layers)


    def forward(self, x):
        x=self.features(x)
        return x


class Generator(nn.Module):
    def __init__(self, sample):
        super(Generator, self).__init__()
        self.size_=((-1),)+ sample.shape
        print(self.size_)
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            nn.Linear(128, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 512),
            nn.Linear(512, sample.size),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(self.size_)
class Discriminator(nn.Module):
    def __init__(self, sample):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(sample.size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img=img.view(img.size()[0], -1)
        validity = self.model(img)

        return validity

