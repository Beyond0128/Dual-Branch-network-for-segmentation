import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# import os
import argparse
import torch
from torch import nn
from loadcloud_data import Cloud_Data
# from loaddata_npy import L8PARCS_Dataset
from prettytable import PrettyTable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from train import train


import warnings
warnings.filterwarnings('ignore')


def main(params, model):
    parser = argparse.ArgumentParser('CloudSeg')

    parser.add_argument('--num_epochs',             type=int,   default=300,        help='None')
    parser.add_argument('--num_classes',            type=int,   default=3,          help='None')
    parser.add_argument('--num_epoch_decay',        type=int,   default=70,         help='change lr')
    parser.add_argument('--checkpoint_step',        type=int,   default=5,          help='save model for every X time')
    parser.add_argument('--validation_step',        type=int,   default=1,          help='check model for every X time')
    parser.add_argument('--batch_size',             type=int,   default=4,          help='None')
    parser.add_argument('--num_workers',            type=int,   default=0,          help='None')
    parser.add_argument('--lr',                     type=float, default=0.0001,     help='None')
    parser.add_argument('--lr_scheduler',           type=int,   default=3,          help='Update the learning rate every X times')
    parser.add_argument('--lr_scheduler_gamma',     type=float, default=0.99,       help='learning rate attenuation coefficient')
    parser.add_argument('--warmup',                 type=int,   default=1,          help='warm up')
    parser.add_argument('--warmup_num',             type=int,   default=1,          help='warm up the number')
    parser.add_argument('--cuda',                   type=str,   default='0',        help='GPU ids used for training')
    parser.add_argument('--beta1',                  type=float, default=0.9,        help='momentum1 in Adam')
    parser.add_argument('--beta2',                  type=float, default=0.999,      help='momentum2 in Adam')
    parser.add_argument('--momentum',               type=float, default=0.9,        help='momentum in SGD')
    parser.add_argument('--miou_max',               type=float, default=0.9,        help='If Miou greater than it ,will be saved and update it')
    parser.add_argument('--dir_name',               type=str,   default='dfn',      help='miou max/tensorvboard/result.txt save path')
    parser.add_argument('--crop_height',            type=int,   default=224,        help='predict on image height')
    parser.add_argument('--crop_width',             type=int,   default=224,        help='predict on image width')
    parser.add_argument('--pretrained_model_path',  type=str,   default=None,       help='None')
    parser.add_argument('--save_model_path',        type=str,   default="./checkpoints/",   help='path to save model')
    parser.add_argument('--data',                   type=str,   default='./clouddata_and_aug/',     help='path of training data')
    parser.add_argument('--log_path',               type=str,   default='./log/',       help='path to save the net_log')
    parser.add_argument('--summary_path',           type=str,   default='./summary/',       help='path to save miou max/tensorvboard/result.txt')
    args = parser.parse_args(params)
    # print(args)

    # 打印参数信息
    
    tb = PrettyTable(['Num', 'Key', 'Value'])  #生成美观的ASCII格式的表格
    args_str = str(args)[10:-1].split(', ')
    # print(args_str)

    for i, key_value in enumerate(args_str):
       key, value = key_value.split('=')[0], key_value.split('=')[1]
       tb.add_row([i + 1, key, value])
    print(tb)

    # 检测是否有参数列表中的 save_model_path参数下对应的路径："./checkpoints/"
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    # 检测是否有参数列表中的 log_path参数下对应的路径："./net_log/"
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    # 检测是否有参数列表中的 summary_path参数下对应的路径："./summary/hr/"
    if not os.path.exists(args.summary_path+args.dir_name):
        os.makedirs(args.summary_path+args.dir_name)
    if not os.path.exists(args.summary_path+args.dir_name+'/checkpoints'):
        os.makedirs(args.summary_path+args.dir_name+'/checkpoints')

    # 创建训练数据集和验证集的（image和label）的路径
    train_path_img = os.path.join(args.data, 'train_image')  # image
    train_path_label = os.path.join(args.data, 'train_label')  # labels

    val_path_img = os.path.join(args.data, 'val_image')
    val_path_label = os.path.join(args.data, 'val_label')

    csv_path = os.path.join(args.data, 'class_dict.csv')

    # 训练数据加载    # train_transform = transforms.Compose([transforms.Resize(512, interpolation=Image.BILINEAR),       #数据预处理
    #                                       transforms.ToTensor()  # 将图片转换为Tensor,归一化至[0,1]
    #                                       ])

    # dataset_train = Cloud_Seg(train_path_img,
    #                           train_path_label,
    #                           csv_path,
    #                           mode='train',
    #                           transform=train_transform)



    dataset_train = Cloud_Data(
        train_path_img,
        train_path_label,
        csv_path,
        mode='train'
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # 验证数据加载
    # val_transform = transforms.Compose([transforms.Resize(512, interpolation=Image.BILINEAR),
    #                                     transforms.ToTensor()
    #                                     ])
    # dataset_val = Cloud_Seg(val_path_img,
    #                         val_path_label,
    #                         csv_path,
    #                         mode='val',
    #                         transform=val_transform)


    dataset_val = Cloud_Data(
        val_path_img,
        val_path_label,
        csv_path,
        mode='val'
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
    )

    # 设置模型和参数
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    # 在 PyTorch 程序开头将其值设置为 True，就可以大大提升卷积神经网络的运行速度。
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, (args.beta1, args.beta2), weight_decay=1e-4)
    # optimizer = RAdam(model.parameters(), args.lr, (args.beta1, args.beta2), weight_decay=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), args.lr, args.momentum, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_scheduler, gamma=args.lr_scheduler_gamma)
    # exp_lr_scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_scheduler_gamma)
    # exp_lr_scheduler = PolynomialLR(optimizer, step_size=1, iter_max=args.num_epochs, power=2.0)
    criterion = nn.CrossEntropyLoss()
    # criterion = SoftmaxFocalLoss()

    # 如果存在预训练好的模型，就加载它
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)

        # loading the part of network params
        pretrained_dict = torch.load(args.pretrained_model_path)
        #给model_dict 赋予运行的那个神经网络（这里是UNet）的参数字典，形式可以查看笔记
        #下面的代码到print，加载预训练模型,并去除需要再次训练的层，注意：需要重新训练的层的名字要和之前的不同。k 神经网络层名字.Weight或者 神经网络层名字.bias
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the model state dict
        model.load_state_dict(model_dict)
        print('Done!')

    # 开始训练
    train(args,
          model,
          csv_path,
          optimizer,
          criterion,
          dataloader_train,
          dataloader_val,
          exp_lr_scheduler
          )


if __name__ == '__main__':
    params = [
        '--num_epochs', '2',
        '--batch_size', '8',
        '--lr', '0.001',
        '--warmup', '1',
        '--lr_scheduler_gamma', '0.95',
        '--lr_scheduler', '3',
        '--miou_max', '0.89',
        '--cuda', '0',
        # '--dir_name', 'resnet50_cat',
        '--dir_name', 'resnet50_demo_1',
        # '--pretrained_model_path', r'D:\miou_0.939055_194.pth'
    ]
    # from xiaorongmodel.resnet50_cat import Resnet50
    # model = Resnet50(backbone='resnet50', pretrained=False, n_class=3)

    from Dual_branch_Network import CvT_CNN_cat_loop_cat_sbr
    model = CvT_CNN_cat_loop_cat_sbr(img_size=224, in_channels=3, num_classes=3).cuda()

    main(params, model)

   