import pdb
import torch
from torch import nn
import pretrainedmodels


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


def get_model(model_name, out_features=1, pretrained="imagenet"):
    return Model(model_name, out_features, pretrained)


if __name__ == "__main__":
    # model_name = "se_resnext50_32x4d_v2"
    # model_name = "nasnetamobile"
    model_name = "resnext101_32x4d_v0"
    classes = 1
    size = 256
    model = Model(model_name, classes, "imagenet")
    image = torch.Tensor(3, 3, size, size) # BN layers need more than one inputs, running mean and std
    # image = torch.Tensor(1, 3, 112, 112)
    # image = torch.Tensor(1, 3, 96, 96)

    output = model(image)
    print(output.shape)
    pdb.set_trace()
