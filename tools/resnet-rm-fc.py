import wget
import torch
import torch.nn as nn
import torchvision.models as models

url_list = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


wget.download(url_list['resnet50'], '../models/resnet50.pth')
"""
    # Download weight & load
    这里下载的只是预训练的权重，后面会将权重加载到网络中
"""
state_dict = torch.load("../models/resnet50.pth")

# Load model
model = models.resnet50()

# Load Weights
model.load_state_dict(state_dict)

"""
    # Remove Final Layer
    ResNet 最后一层为 FC 全连接层
"""
model = nn.Sequential(*(list(model.children())[:-1]))
print(model)
print(model.state_dict())

example = torch.rand(1,3,224,224)
traced_script_module = torch.jit.trace(model, example)

output = traced_script_module(torch.ones(1,3,224,224))
print(type(output), output[0,:5],output.shape)

# traced_script_module.save("./models/resnet50-rm-fc.pth")