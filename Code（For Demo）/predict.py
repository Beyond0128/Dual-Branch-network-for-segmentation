import cv2
from imgaug import augmenters as iaa
from PIL import Image
import torchvision.transforms as transforms
from torch import nn

from utils import get_label_info, reverse_one_hot, colour_code_segmentation
import numpy as np
import os
import torch


def predict_on_image(model, args, epoch, csv_path):
    # pre-processing on image

    test_path = os.path.join(args.demo_root, args.demo_name)
    test_list = os.listdir(test_path)

    for i in test_list:
        image = cv2.imread(os.path.join(test_path, i), -1)  # 读入一张图片数据，图片为BGR形式的数组
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR图片数组 转换为RGB图片数组
        resize = iaa.Resize({'height': args.crop_height, 'width': args.crop_width})  # 剪裁尺寸
        resize_det = resize.to_deterministic()  # 保持坐标和图像同步改变，而不是随机
        image = resize_det.augment_image(image)  # 保存变换后的图片
        image = Image.fromarray(image).convert('RGB')  # 将数组转换为RGB图片
        image = transforms.ToTensor()(image).unsqueeze(0)  # 标准化后，在第0维增加一个维度
        # read csv label path
        label_info = get_label_info(csv_path)
        # predict
        model.eval()
        predict = model(image.cuda())[0]

        w = predict.size()[-1]
        c = predict.size()[-3]
        predict = predict.resize(c, w, w)
        predict = reverse_one_hot(predict)  # 此处返回的是HWC 最后一维处最大值的序号，用来判断像素点的颜色

        predict = colour_code_segmentation(np.array(predict.cpu()), label_info)  # 对每一个像素点进行分类，得到分类后的图片数据
        predict = cv2.resize(np.uint8(predict),
                             (args.crop_height, args.crop_width))  # 数据类型转换为unit8，， uint8为无符号整型数据, 范围是从0–255
        save_path = f'/{i[:-4]}_epoch_%d.png' % (epoch)
        cv2.imwrite('demo/' + args.save_model_path + save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
        # cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))

def predict1(model, demo_path, size, load_state_dict, csv_path, save_path):
    test_list = os.listdir(demo_path)
    for i in test_list:
        image = cv2.imread(demo_path + i, -1)  # 读入一张图片数据，图片为BGR形式的数组
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR图片数组 转换为RGB图片数组

        image = Image.fromarray(image).convert('RGB')  # 将数组转换为RGB图片
        image = transforms.ToTensor()(image).unsqueeze(0)  # 标准化后，在第0维增加一个维度
        # read csv label path
        label_info = get_label_info(csv_path)

        model.load_state_dict(torch.load(load_state_dict))
        model.eval()
        predict = model(image)

        w = predict.size()[-1]
        c = predict.size()[-3]
        predict = predict.resize(c, w, w)
        predict = reverse_one_hot(predict)  # 此处返回的是HWC 最后一维处最大值的序号，用来判断像素点的颜色

        predict = colour_code_segmentation(np.array(predict.cpu()), label_info)  # 对每一个像素点进行分类，得到分类后的图片数据
        predict = cv2.resize(np.uint8(predict), (size, size))  # 数据类型转换为unit8，， uint8为无符号整型数据, 范围是从0–255
        save_path1 = f'/{i[:-4]}.png'
        cv2.imwrite(save_path + save_path1, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    # from v2xiaorongmodel.pvtv2n_MSConv_CNN_MGAM_GAFM_43 import pvt_tiny
    # model = pvt_tiny()
    # from duibimodel.Dual_branch_Network import CvT_CNN_cat_loop_cat_sbr
    # model = CvT_CNN_cat_loop_cat_sbr(img_size=224, in_channels=3, num_classes=2)
    # from duibimodel.pyramid_attention_network import PAN
    # model = PAN(backbone='resnet50', pretrained=False, n_class=3)
    # from duibimodel.pspnet import PSPNet
    # model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=1, use_ppm=True, pretrained=False)
    # from duibimodel.LinkNet import LinkNet
    # model = LinkNet(classes=2)
    # from duibimodel.bisenetv2 import BiSeNetV2
    # model = BiSeNetV2(3)
    # from duibimodel.cj import cj
    # model = cj()
    # from duibibackbone.PVT_u import pvt_ti
    # model = pvt_ti()
    # from duibibackbone.CvT_u import CvT
    # model = CvT(img_size=224, in_channels=3, num_classes=2)


    ################ 1 ############### 导入对应的模型
    # # 使用Dual_branch_network训练模型
    # from Dual_branch_Network import CvT_CNN_cat_loop_cat_sbr
    # # model = CvT_CNN_cat_loop_cat_sbr(img_size=224, in_channels=3, num_classes=3).cuda()
    # model = CvT_CNN_cat_loop_cat_sbr(img_size=224, in_channels=3, num_classes=3)

    # # 使用CGNet下载模型训练
    # from cgnet import CGNet
    # # model = CGNet(img_size=224, in_channels=3, num_classes=3).cuda()
    # # model = CGNet(3)
    # model = CGNet(classes=3, M=3, N=9)

    # # PAN模型
    # from pyramid_attention_network import PAN
    # model = PAN(backbone='resnet50', pretrained=True, n_class=3)

    # # PSPNet模型
    # from pspnet import PSPNet
    # model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=3, zoom_factor=1, use_ppm=True, pretrained=False)

    # LinkNet模型
    from LinkNet import LinkNet
    model = LinkNet(classes=3)

   # 导入图片
    demo_path = './demo_img/'



    ############### 2 ################ 权重训练模型
    load_state_dict = './checkpoints/LinkNet_miou_0.903877.pth'  # LinkNet

    size = 224
    csv_path = './clouddata_and_aug/class_dict.csv'


    ############# 3 ########### 存放生成照片的地方
    save_path = './picture_img_LinkNet'  # LinkNet


    predict1(model=model, demo_path=demo_path, load_state_dict=load_state_dict, size=size, csv_path=csv_path,
             save_path=save_path)

