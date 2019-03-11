import pdb
import torch
from torch import nn
import pretrainedmodels


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class Model(nn.Module):
    def __init__(self, model_name, out_features, pretrained="imagenet"):
        super(Model, self).__init__()

        if model_name in ["se_resnet50", "se_resnext50_32x4d"]:
            model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained=pretrained
            )
            blocks = list(model.children())
            self.backbone = nn.Sequential(
                *list(blocks[0].children())[:-1]
            )  # remove maxpool
            self.classifier = nn.Sequential(
                nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
                Flatten(),
                nn.Linear(in_features=2048, out_features=out_features, bias=True),
            )
            print("Make sure to use 3x112x112 shaped input images")

        elif model_name in ["se_resnet50_v2", "se_resnext50_32x4d_v2"]:
            """Takes in 3x112x112"""
            model_name = model_name[:-3]  # clip the _v2 part
            model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained=pretrained
            )
            # remove Maxpool, remove last two layers (avg_pool, linear_layer
            blocks = list(model.children())
            base = list(blocks[0].children())[:-1] + blocks[1:-2]
            self.backbone = nn.Sequential(*base)
            # add custom layers, designed for 3x96x96 input image size
            self.classifier = nn.Sequential(
                nn.ConstantPad2d((0, 1, 0, 1), 0),
                nn.AvgPool2d(kernel_size=7, stride=1, padding=0),
                Flatten(),
                nn.Dropout(0.3),
                nn.Linear(in_features=2048, out_features=out_features, bias=True),
            )
            print("Make sure to use 3x96x96 shaped input images")

        elif model_name in ["se_resnet50_v3", "se_resnext50_32x4d_v3"]:
            """Takes in 3x96x96"""
            model_name = model_name[:-3]  # clip the _v3 part
            model = pretrainedmodels.__dict__[model_name](
                num_classes=1000, pretrained=pretrained
            )
            # remove Maxpool, remove last two layers (avg_pool, linear_layer
            blocks = list(model.children())
            base = list(blocks[0].children())[:-1] + blocks[1:-2]
            self.backbone = nn.Sequential(*base)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                Flatten(),
                nn.Dropout(0.1),
                nn.Linear(in_features=2048, out_features=out_features, bias=True),
            )
        elif model_name == "nasnetamobile_v2":
            """ Takes in 3x224x224"""
            model_name = model_name[:-3]
            self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=pretrained)
            self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    Flatten(),
                    nn.Dropout(0.5),
                    nn.Linear(in_features=1056, out_features=out_features, bias=True)
                    )

    def forward(self, x):
        x = self.model.features(x) # only backbone features
        x = self.classifier(x)
        return x

def get_model(model_name, out_features=1, pretrained="imagenet"):
    if model_name == "nasnetamobile":
        model = pretrainedmodels.__dict__[model_name](
                    num_classes=1000, pretrained=pretrained
        )
        model._modules["last_linear"] = nn.Linear(in_features=1056, out_features=out_features, bias=True)
        return model

if __name__ == "__main__":
    #model_name = "se_resnext50_32x4d_v2"
    model_name = "nasnetamobile"

    model = Model(model_name, 1, "imagenet")
    image = torch.Tensor(1, 3, 224, 224)
    # image = torch.Tensor(1, 3, 112, 112)
    #image = torch.Tensor(1, 3, 96, 96)

    output = model(image)
    print(output.shape)
    pdb.set_trace()
