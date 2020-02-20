import torchvision
import torch
import numpy as np
import os
import time
import torch.nn as nn
import matplotlib.pyplot as plt

BATCH_SIZE = 32
LEARING_RATE = 0.00005
CUDA = torch.cuda.is_available()
EPOCH = 200
N_CLASS = 21
GPU_USE = [0, 1]
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
TOT = 0

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


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class FCN16(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=100),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),

            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.feature2 = nn.Sequential(
            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        self.feature3 = nn.Sequential(
            # conv5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )

        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(True),
            nn.Dropout2d()
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(True),
            nn.Dropout2d()
        )
        self.classifier = nn.Conv2d(4096, N_CLASS, 1)
        self.classifier2 = nn.Conv2d(512, N_CLASS, 1)
        self.classifier2.weight = nn.Parameter(torch.zeros(self.classifier2.weight.data.size()), requires_grad=True)
        self.classifier2.bias = nn.Parameter(torch.zeros(self.classifier2.bias.data.size()), requires_grad=True)

        self.upscore2 = nn.ConvTranspose2d(N_CLASS, N_CLASS, 4, stride=2, bias=False)
        self.upscore16 = nn.ConvTranspose2d(N_CLASS, N_CLASS, 32, stride=16, bias=False)
        self.init_vgg_para()

    def forward(self, x):
        pool3 = self.feature(x)
        pool4 = self.feature2(pool3)
        pool5 = self.feature3(pool4)
        after_fc6 = self.fc6(pool5)
        after_fc7 = self.fc7(after_fc6)
        after_class = self.classifier(after_fc7)
        after_class_pool4 = self.classifier2(pool4)

        after_up = self.upscore2(after_class)
        after_class_pool4 = after_class_pool4[:, :, 5:5 + after_up.size()[2], 5:5 + after_up.size()[3]].contiguous()
        after_class_pool4 = after_class_pool4 + after_up
        after_class_pool4 = self.upscore16(after_class_pool4)
        h = after_class_pool4[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]].contiguous()
        return h

    def init_vgg_para(self):
        tmp_para = torch.load("/output/model32/32-epoch32.pkl")
        tmp_para = tmp_para.module.cpu()
        for l1, l2 in zip([*zip(tmp_para.feature), *zip(tmp_para.feature2), *zip(tmp_para.feature3)], [*zip(self.feature), *zip(self.feature2), *zip(self.feature3)]):
            l1 = l1[0]
            l2 = l2[0]
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                if l1.weight.size() != l2.weight.size():
                    print("error")
                    exit(0)
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
                print(l1)
                print(l2)
        self.fc6[0].weight.data = tmp_para.fc6[0].weight.data
        self.fc7[0].weight.data = tmp_para.fc7[0].weight.data
        self.fc6[0].bias.data = tmp_para.fc6[0].bias.data
        self.fc7[0].bias.data = tmp_para.fc7[0].bias.data
        print(self.fc6[0].weight)
        self.classifier.weight.data = tmp_para.classifier.weight.data
        self.classifier.bias.data = tmp_para.classifier.bias.data
        self.upscore2.weight = nn.Parameter(get_upsampling_weight(N_CLASS, N_CLASS, 4), requires_grad=True)
        self.upscore16.weight = nn.Parameter(get_upsampling_weight(N_CLASS, N_CLASS, 32), requires_grad=True)


def cal_acc(input, target):
    input = torch.argmax(input, dim=1)
    sum = torch.eq(input, target).sum()
    all = input.size()[1] * input.size()[2] * input.size()[0]
    global TOT
    torch.save((input.clone().cpu(), target.clone().cpu()), "/output/PNG/out{}.pkl".format(TOT % 10))
    TOT += 1

    print(torch.max(input), torch.max(target), input.size(), target.size())
    return float(sum) / all


def cal_background(input, target):
    input = torch.eq(torch.argmax(input, dim=1), 0).sum()
    return int(input), int(target.size()[0] * target.size()[1] * target.size()[2])


class mySBD(torchvision.datasets.SBDataset):
    def __init__(self):
        super().__init__(mode='segmentation', root="/data/benchmark_RELEASE/dataset")

    def __getitem__(self, item):
        img, tar = super().__getitem__(item)
        return train_transform(img), target_transform(tar)


if __name__ == '__main__':
    train_set = mySBD()

    train_set_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True
    )

    model = FCN16()  # type: nn.Module

    if CUDA:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=GPU_USE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARING_RATE)

    tot = len(train_set)
    now_has_run = 0
    start_time = time.time()
    end_time = time.time()
    print(tot)
    for epoch in range(EPOCH):
        for batch, (image, target) in enumerate(train_set_loader):
            if CUDA:
                image = image.cuda()
                target = target.cuda()
            now_has_run += BATCH_SIZE

            target = (target * 255).long().squeeze(dim=1)
            # optimizer.zero_grad()
            model.zero_grad()
            output = model(image)

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            if batch % 50 == 0:
                end_time = time.time()
                print("cost {}s, TOT is ".format(end_time - start_time), TOT)
                print("loss is ", float(loss))
                print("acc is ", cal_acc(output, target), "  mean acc ", cal_background(output, target))
                print("you have run {}%".format(now_has_run / tot * 100))
                print("\n\n")
                start_time = time.time()
        if (epoch + 1) % 10 == 0:
            torch.save(model, "/output/model/pth{}.pkl".format(epoch))

