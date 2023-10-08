# PyTorch Освещая глубокое обучение (Питер 2022[+1;)])

#python3 -m venv envtorch
#source envtorch/bin/activate
#pip install --upgrade pip

#https://pytorch.org/get-started/locally/
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

#deactivate

'''
def describe(x): # для вывода различных характеристик тензора x, например типа тензора, его размерности и содержимого
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}\n".format(x))


import torch
print(f'torch.__version__ = {torch.__version__}')

describe(torch.Tensor(2, 3))
describe(torch.rand(2, 3)) # случайное равномерное распределение
describe(torch.randn(2, 3)) # случайное нормальное распределение

describe(torch.zeros(2, 3))
x = torch.ones(2, 3)
describe(x)
x.fill_(5)
describe(x)

x = torch.Tensor([[1, 2, 3],[4, 5, 6]])
describe(x)

import numpy as np
npy = np.random.rand(2, 3)
describe(torch.from_numpy(npy))

x = torch.tensor([[1, 2, 3],
[4, 5, 6]], dtype=torch.int64)
describe(x)

x = x.float()
describe(x)
describe(x + x)
describe(torch.transpose(x.add(x), 0, 1))
describe(x[0, 1])

describe(torch.cat([x, x], dim=0))
describe(torch.cat([x, x], dim=1))
'''

from torchvision import models
#print(dir(models))

resnet = models.resnet101(pretrained=True)
'''UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 
and may be removed in the future. The current behavior is equivalent to passing 
`weights=ResNet101_Weights.IMAGENET1K_V1`. 
You can also use `weights=ResNet101_Weights.DEFAULT` to get the most up-to-date weights.'''
print(resnet)

from torchvision import transforms
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(
mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225]
)])


from PIL import Image
img = Image.open("bobby.png")


img_t = preprocess(img)

import torch
batch_t = torch.unsqueeze(img_t, 0)

resnet.eval()
out = resnet(batch_t)

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

_, index = torch.max(out, 1)

percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())

_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])

'''
golden retriever 82.66133880615234

[('golden retriever', 82.66133880615234),
 ('Great Pyrenees', 8.645225524902344), 
('Labrador retriever', 4.516740322113037),
 ('English setter', 1.3028920888900757),
 ('tennis ball', 0.5473359227180481)]
'''
