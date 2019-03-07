import pdb
import torch
import pretrainedmodels


def Model(model_name, out_features, pretrained='imagenet'):
    if model_name == "se_resnet50" or model_name == "se_resnext50_32x4d":
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=pretrained)
        blocks = list(model.children())
        head = torch.nn.Sequential(*list(blocks[0].children())[:-1]) # remove maxpool
        model._modules['layer0'] = head
        model._modules['last_linear'] = torch.nn.Linear(in_features=2048, out_features=out_features, bias=True)
        return model

if __name__ == "__main__":
    model = Model('se_resnet50', 1, 'imagenet')
    image = torch.Tensor(1, 3, 112, 112)
    output = model(image)
    print(output.shape)
    pdb.set_trace()
