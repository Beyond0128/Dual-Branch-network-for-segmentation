import glob, os, torch
import random
from PIL import Image
from utils import *
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F 
import matplotlib.pyplot as plt


def imagestransforms(img, label):
    img = transforms.Resize(224, interpolation=Image.BILINEAR)(img)
    label = transforms.Resize(224, interpolation=Image.NEAREST)(label)

    # 以0.2的概率进行,随机旋转，水平翻转，垂直翻转操作
    if random.random() < 0.2:
        img = F.hflip(img)
        label = F.hflip(label)

    if random.random() < 0.2:
        img = F.vflip(img)
        label = F.vflip(label)

    if random.random() < 0.2:
        angle = transforms.RandomRotation.get_params([-180, 180])
        img = img.rotate(angle, resample=Image.BILINEAR)
        label = label.rotate(angle, resample=Image.NEAREST)

    img = F.to_tensor(img).float()
    return img, label


class Cloud_Data(torch.utils.data.Dataset):
    def __init__(self, img_path, label_path, csv_path, mode='train'):
        '''
        继承Dataset类,加载数据
        :param img_path: 图片路径
        :param label_path: 标签路径
        :param csv_path: colormap(每一个类的RGB值)
        :param mode:
        :param transform: 数据增强
        '''
        super().__init__()
        self.mode = mode
        self.img_list = glob.glob(os.path.join(img_path,  '*.png'))  # 读取所有png文件
        self.label_list = glob.glob(os.path.join(label_path, '*.png'))

        self.label_colormap = get_label_colormap(csv_path)


    def __getitem__(self, index):
        '''
        把rgb图转化为标签图
        :param index:
        :return: torch类型的img和label
        '''
        img = Image.open(self.img_list[index]).convert('RGB')  # 这里.convert('RGB')可加可不加 把图片转化成RGB模式
        label = Image.open(self.label_list[index]).convert('RGB')

        if self.mode == 'train':
            img , label = imagestransforms(img, label)
        else:
            img = transforms.Resize(224, interpolation=Image.BILINEAR)(img)
            img = transforms.ToTensor()(img).float()
            label = transforms.Resize(224, interpolation=Image.NEAREST)(label)

        label = image2label(label, self.label_colormap)
        label = torch.from_numpy(label)

        return img, label

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # transform = transforms.Compose([transforms.Resize(512, interpolation=Image.BILINEAR),
    #                                 transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
    #                                 ])
    data = Cloud_Data(r'C:\Users\ggw\ZDS\ZDS\L7_code\cloud_data\train\cloudshadow',
                     r'C:\Users\ggw\ZDS\ZDS\L7_code\cloud_data\train\labels',
                     r'C:\Users\ggw\ZDS\ZDS\L7_code\cloud_data\class_dict.csv',
                     mode='train',
                     )
    print(len(data))
    dataloader_test = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    for i, (img, label) in enumerate(data):
        print(img)
        print(img.shape)
        print(max(img.flatten()))
        print(label)
        if i >= 0:
            break

    img = Image.open(r'C:\Users\ggw\ZDS\ZDS\L7_code\cloud_data\train\cloudshadow\0.png').convert('RGB')
    label = Image.open(r'C:\Users\ggw\ZDS\ZDS\L7_code\cloud_data\train\labels\0.png').convert('RGB')

    def imagestransforms(img, label):
        img = transforms.Resize(512, interpolation=Image.BILINEAR)(img)
        label = transforms.Resize(512, interpolation=Image.NEAREST)(label)

        # 以0.5的概率进行,随机旋转，水平翻转，垂直翻转操作
        random.seed(32)
        a = random.random()
        print("Random number with seed a : ", a)
        random.seed(16)
        b = random.random()
        print("Random number with seed b : ", b)
        random.seed(3)
        c = random.random()
        print("Random number with seed c : ", c)
        if a < 0.2:
            img = F.hflip(img)
            label = F.hflip(label)
        if b < 0.2:
            img = F.vflip(img)
            label = F.vflip(label)
        if c < 0.2:
            angle = transforms.RandomRotation.get_params([-180, 180])
            img = img.rotate(angle, resample=Image.BILINEAR)
            label = label.rotate(angle, resample=Image.NEAREST)


        return img, label

    img,label = imagestransforms(img,label)
    plt.imshow(img)
    plt.show()
    plt.imshow(label)
    plt.show()



