import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os


BATCH_SIZE = 32
CUDA = torch.cuda.is_available()
N_CLASS = 21
GPU_USE = [0, 1]
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
TOT = 0


def cal_acc(self, input, target):
    x = torch.eq(input, target).sum()
    return float(x)/float(torch.numel(target))


def cal_mean_acc(self, input, target):
    corr = 0
    tot = 0
    for i in range(N_CLASS):
        tmp = torch.eq(target, ((torch.ones(target.size())*i).long()).cuda())
        i_num = int(tmp.sum())
        ok = int(torch.eq(torch.where(tmp, input, (-torch.ones(tmp.size()).long()).cuda()), i).sum())
        if i_num == 0:
            continue
        tot += 1
        corr += ok/i_num
    return corr / tot


def cal_iu(self, input, target):
    corr = 0
    tot = 0
    for i in range(N_CLASS):
        input_tmp = torch.eq(input, i)
        target_tmp = torch.eq(target, i)
        ok = int((input_tmp | target_tmp).sum())
        input_tmp = torch.where(input_tmp, input_tmp.long(), (-torch.ones(target.size()).long()).cuda())
        target_tmp = torch.where(target_tmp, target_tmp.long(), (torch.zeros(target.size()).long()).cuda())
        cort = int(torch.eq(input_tmp, target_tmp).sum())
        if ok == 0 and cort == 0:
            continue
        tot += 1
        corr += cort / ok
    return corr / tot


normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
target_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),

])
train_transform = torchvision.transforms.Compose([
    target_transform,
    normalize
])


class mySBD(torchvision.datasets.SBDataset):
    def __init__(self):
        super().__init__(mode='segmentation', root="/code/benchmark_RELEASE/dataset", image_set='val')

    def __getitem__(self, item):
        img, tar = super().__getitem__(item)
        return train_transform(img), target_transform(tar)


model = torch.load("/code/model16/16model-199.pkl")

model = model.cuda()
# model = torch.nn.DataParallel(model, device_ids=GPU_USE, output_device=GPU_USE)
model.eval()
test_set = mySBD()
test_set_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=True,
    num_workers=8,
    pin_memory=True
)

tot =len(test_set)

pixel_acc = 0
pixel_mean_acc = 0
mean_iu = 0
print(tot)
for i, (image, target) in enumerate(test_set_loader):
    image = image.cuda()
    target = target.cuda()
    target = (target * 255).long().squeeze(dim=1)
    output = model(image)
    output = torch.argmax(output, dim=1)
    pixel_acc += model.cal_acc(output, target)
    pixel_mean_acc += model.cal_mean_acc(output, target)
    mean_iu += model.cal_iu(output, target)
    if i % 10 == 0:
        num = i + 1
        print("you have run{}%".format(num * BATCH_SIZE / tot * 100))
        print("acc is {}            mean acc is {}".format(pixel_acc / num, pixel_mean_acc / num))
        print("mean iu is ", mean_iu / num)
        print("\n\n")


