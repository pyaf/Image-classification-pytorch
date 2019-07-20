import pdb
import torch
from torch import nn
import pretrainedmodels
from efficientnet_pytorch import EfficientNet

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz):
        "Output will be 2*sz or 2 if sz is None"
        super(AdaptiveConcatPool2d, self).__init__()
        self.output_size = sz or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x):
        #pdb.set_trace()
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Model(nn.Module):
    def __init__(self, model_name, out_features, pretrained="imagenet"):
        super(Model, self).__init__()
        if model_name in ["se_resnet50"]:
            self.model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained=pretrained
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Dropout(0.3),
                nn.Linear(in_features=2048, out_features=1024, bias=True),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(in_features=1024, out_features=out_features, bias=True),
            )
        elif model_name == "se_resnet50_v0":
            model_name = "se_resnet50"
            self.model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained=pretrained
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Dropout(0.3),
                nn.Linear(in_features=2048, out_features=out_features, bias=True),
            )
        elif model_name in ["densenet121"]:
            self.model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained=pretrained
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Dropout(0.3),
                nn.Linear(in_features=1024, out_features=out_features, bias=True),
            )

        elif model_name == "resnext101_32x4d":
            self.model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained=pretrained
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Dropout(0.5),
                nn.Linear(in_features=2048, out_features=out_features, bias=True),
            )
        elif model_name == "resnext101_32x4d_v0":
            model_name = "resnext101_32x4d"
            self.model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained=pretrained
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                Flatten(),
                nn.BatchNorm1d(2048),
                nn.Dropout(p=0.25),
                nn.Linear(in_features=2048, out_features=2048, bias=True),
                nn.ReLU(),
                nn.BatchNorm1d(2048),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=2048, out_features=out_features, bias=True),
             )

        elif model_name == "resnext101_32x4d_v1":
            model_name = "resnext101_32x4d"
            self.model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained=pretrained
            )
            self.classifier = nn.Sequential(
                AdaptiveConcatPool2d(1),
                Flatten(),
                nn.BatchNorm1d(4096),
                nn.Dropout(p=0.25),
                nn.Linear(in_features=4096, out_features=2048, bias=True),
                nn.ReLU(),
                nn.BatchNorm1d(2048),
                nn.Dropout(p=0.5),
                nn.Linear(in_features=2048, out_features=out_features, bias=True),
             )

    def forward(self, x):
        #pdb.set_trace()
        x = self.model.features(x)  # only backbone features
        x = self.classifier(x)
        return x


def resnext101_32x16d(out_features):
    '''[1]'''
    model = torch.hub.load(
                'facebookresearch/WSL-Images',
                'resnext101_32x16d_wsl'
    )
    for params in model.parameters():
        params.requires_grad = False

    model.fc = nn.Linear(in_features=2048, out_features=out_features, bias=True)
    # every new layer added, has requires_grad = True

    return model


def efficientNet(name, out_features):
    '''name like: `efficientnet-b5`
    [2]
    '''

    model = EfficientNet.from_pretrained(name, num_classes=out_features)

    for params in model.parameters():
        params.requires_grad = False

    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

    return model


def get_model(model_name, out_features=1, pretrained="imagenet"):
    if model_name == "resnext101_32x16d":
        return resnext101_32x16d(out_features)
    elif model_name.startswith("efficientnet"):
        return efficientNet(model_name, out_features)
    return Model(model_name, out_features, pretrained)


if __name__ == "__main__":
    # model_name = "se_resnext50_32x4d_v2"
    # model_name = "nasnetamobile"
    #model_name = "resnext101_32x4d_v0"
    model_name = "efficientnet-b5"
    classes = 1
    size = 256
    model = get_model(model_name, classes, "imagenet")
    image = torch.Tensor(3, 3, size, size) # BN layers need more than one inputs, running mean and std
    # image = torch.Tensor(1, 3, 112, 112)
    # image = torch.Tensor(1, 3, 96, 96)

    output = model(image)
    print(output.shape)
    pdb.set_trace()


''' footnotes

[1]: model.avgpool is already AdapativeAvgPool2d, and model's forward method handles flatten and stuff. So here I'm just adding a trainable the last fc layer, after few epochs the model's all layers will be set required_grad=True
Apart from that this model is trained on instagram images, remove imagenet mean and std, only gotta divide by 255, so mean=0,std=1

[2]: efficientnet models are trained on imagenet, so make sure mean and std are of imagenet.
'''
