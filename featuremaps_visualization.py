
import copy
from torchvision import datasets, transforms, models
import torch                                                  #tensor related operations
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from ResNet50_Deeplabv3 import resnet50_DV
from resnet18 import resnet18_DV
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


testing_transforms = transforms.Compose([
        # transforms.Resize(256),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 以下对于rest net18可以跑通feature map绘制
model1 = resnet50_DV()
# model1 = resnet18_DV()
# print(model1)
# print(model)
pretrain_weights_path = "./model_new_all.pth"
# pretrain_weights_path = "./resnet50meiyoue_model_onlyweigths.pth"
model1.load_state_dict(torch.load(pretrain_weights_path))
# target_layers = [model.backbone.RSU1]
# we will save the conv layer weights in this list
model_weights =[]
#we will save the 49 conv layers in this list
conv_layers = []

counter = 0
#
print(f"Total convolution layers: {counter}")
print("conv_layers")
model = model1.to(device)

# print(conv_layers)

#
# image_dir = "./FFFFF/airplane18.tif"
image = Image.open(str('./FFFFF/tenniscourt23.tif'))

image = testing_transforms(image)
print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
print(f"Image shape after: {image.shape}")

image = image.to(device)
ret_bef, ret_aft, ret_FPN, ret_reduction = model(image)
# ret_before2, ret_after2,= model(image)

processed = []
for feature_map in ret_FPN:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map,0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
for fm in processed:
    print(fm.shape)


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(100, 100))
for i in range(len(processed)):
    a = fig.add_subplot(5, 4, i+1)
    # imgplot = plt.imshow(processed[i], cmap='gray', interpolation='nearest')
    imgplot = plt.imshow(processed[i],interpolation='nearest')
    a.axis("off")
    # a.set_title(names[i].split('(')[0], fontsize=30)
plt.savefig(str('./acc/2020.jpg'), bbox_inches='tight')



model = resnet50()
# print(model)
pretrain_weights_path = "./resnet50meiyoue_model_onlyweigths.pth"
# pretrain_weights_path = "./resnet50meiyoue_model_onlyweigths.pth"
model.load_state_dict(torch.load(pretrain_weights_path))
# target_layers = [model.backbone.RSU1]
model_layer = list(model.backbone.children())

