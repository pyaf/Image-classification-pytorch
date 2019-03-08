import pdb
import torch
import pretrainedmodels


class Model(torch.nn.Module):
    def __init__(self, model_name, out_features, pretrained='imagenet'):
        super(Model, self).__init__()
        if model_name == "se_resnet50" or model_name == "se_resnext50_32x4d":
            model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=pretrained)
            # remove Maxpool, remove last two layers (avg_pool, linear_layer
            blocks = list(model.children())
            base = list(blocks[0].children())[:-1] + blocks[1:-2]
            self.backbone = torch.nn.Sequential(*base)
            # add custom layers, designed for 3x96x96 input image size
            self.pad = torch.nn.ConstantPad2d((0,1,0,1), 0)
            self.avg_pool = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            self.dropout = torch.nn.Dropout(0.3)
            self.linear = torch.nn.Linear(in_features=2048, out_features=out_features, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pad(x)
        x = self.avg_pool(x)
        x = x.view(-1, 2048)
        x = self.dropout(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    model = Model('se_resnet50', 1, 'imagenet')
    #image = torch.Tensor(1, 3, 112, 112)
    image = torch.Tensor(1, 3, 96, 96)
    output = model(image)
    print(output.shape)
    pdb.set_trace()
