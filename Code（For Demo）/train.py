import os
import tqdm
import time
from tensorboardX import SummaryWriter
from utils import *
from evaluation import *
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score  # input(GT, SR)
from loss import SoftmaxFocalLoss
from PIL import Image
import matplotlib.pyplot as plt


def val(args, model, criterion, num_classes, csv_path, dataloader_val, epoch, loss_train_mean, iou_train_mean, writer):
    print('Val...')
    start = time.time()
    # model.eval()主要用于通知dropout层和batchnorm层在train和val模式间切换
    # 在train模式下，dropout网络层会按照设定的参数p设置保留激活单元的概率（保留概率=p); batchnorm层会继续计算数据的mean和var等参数并更新。
    # 在val模式下，dropout层会让所有的激活单元都通过，而batchnorm层会停止计算和更新mean和var，直接使用在训练阶段已经学出的mean和var值。
    # 该模式不会影响各层的gradient计算行为，即gradient计算和存储与training模式一样，只是不进行反传（backprobagation）
    # 而with torch.zero_grad()则主要是用于停止autograd模块的工作，以起到加速和节省显存的作用，具体行为就是停止gradient计算，
    # 从而节省了GPU算力和显存，但是并不会影响dropout和batchnorm层的行为。
    with torch.no_grad():             #禁止所有的需要梯度的张量进行梯度计算
        model.cuda()                  #将模型移动到GPU进行运算
        model.eval()                  #将模型切换成评估模式
        loss_record = []              #用于记录训练过程中的损失函数值
        hist = torch.zeros((num_classes, num_classes))               #创建用于储存分类问题的混淆矩阵
        evaluator = Evaluator(num_classes)  # epsilon用来防止0作为被除数   计算指标

        for i, (img, label) in enumerate(dataloader_val):        #遍历验证集中的每个样本并放在GPU运算
            img, label = img.cuda(),  label.cuda()

            output = model(img)
            loss = criterion(output, label)              #计算损失

            # 计算评价指标   获取模型对当前批次验证集样本的预测结果
            predict = torch.argmax(output, 1)
            # 将训练结果放到cpu上去计算
            pre = predict.data.view(-1).cpu().numpy()
            lab = label.data.view(-1).cpu().numpy()

            # 计算hist混淆矩阵
            hist += fast_hist(pre, lab, num_classes)

            # 累计loss
            loss_record.append(loss.item())

        # 在save_path路径下显示每个epoch的预测图片
        if (epoch+1) % 1 == 0:
            predict_on_image(model,
                             args.crop_height,
                             args.crop_width,
                             csv_path,
                             read_path="demo/0000_0_4.png",
                             save_path='demo/epoch_%d.png' % (epoch+1))  # _npy

        # compute average value
        loss_val_mean = np.mean(loss_record)  # 发现cpu快
        pa = evaluator.pixel_accuracy(hist)
        recall = evaluator.recall(hist)
        precision = evaluator.precision(hist)
        f1 = evaluator.f1_score(hist)
        miou = evaluator.mean_intersection_over_union(hist)
        fwiou = evaluator.frequency_weighted_intersection_over_union(hist)

        str_ = ("%15.5g;" * 10) % (epoch+1, loss_train_mean, loss_val_mean, pa, recall, precision, f1, iou_train_mean, miou, fwiou)   #组成字符串，可以用于输出到屏幕或者写入文件，以记录训练过程中的各种指标

        # 将验证的结果记录在result.txt以及summary对应文件下的txt文件中
        with open('result.txt', 'a') as f:
            f.write(str_ + '\n')
        with open(os.path.join(args.summary_path, args.dir_name, '')+args.dir_name+'_result.txt', 'a') as f:
            f.write(str_ + '\n')

        # 打印出改正的验证索引
        print('Val_loss:    {:}'.format(loss_val_mean))
        print('PA:          {:}'.format(pa))
        print('Recall:      {:}'.format(recall))
        print('Precision:   {:}'.format(precision))
        print('F1:          {:}'.format(f1))
        print('Miou:        {:}'.format(miou))
        print('FWiou:       {:}'.format(fwiou))
        print('Eval_time:   {:}s'.format(time.time() - start))

        # 写进log
        writer.add_scalars('loss', {'train_loss': loss_train_mean, 'val_loss': loss_val_mean}, epoch+1)  # 读写费时
        writer.add_scalars('miou', {'train_miou': iou_train_mean, 'val_miou': miou}, epoch+1)
        writer.add_scalar('{}_Loss'.format('val'), loss_val_mean, epoch+1)
        writer.add_scalar('{}_Pa'.format('val'), pa, epoch+1)
        writer.add_scalar('{}_Recall'.format('val'), recall, epoch+1)
        writer.add_scalar('{}_Precision'.format('val'), precision, epoch+1)
        writer.add_scalar('{}_F1'.format('val'), f1, epoch+1)
        writer.add_scalar('{}_Miou'.format('val'), miou, epoch+1)
        writer.add_scalar('{}_FWiou'.format('val'), fwiou, epoch + 1)
        # 使用直方图可视化网络中参数的分布情况
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.cuda().data.cpu().numpy(), epoch+1)

        # 返回主要的验证索引
        return miou


def train(args, model, csv_path, optimizer, criterion, dataloader_train, dataloader_val, exp_lr_scheduler):
    print("Train...")
    miou_max = args.miou_max  # miou_max为下限
    miou_max_save_path = args.summary_path + args.dir_name  # miou_max保存路径
    miou_max_save_model_dict = None  # miou_max模型参数
    writer = SummaryWriter(logdir=os.path.join(args.summary_path, args.dir_name))  # 给可视化定义一个安放的路径

    # 打印表头
    s = ("%15s;" * 10) % ("epoch", "train_loss", "val_loss", "PA", "Recall", "Precision", "F1", "Train_Miou", "Val_Miou", "FWiou")
    with open('result.txt', 'a') as f:
        f.write(s + '\n')
    with open(os.path.join(args.summary_path, args.dir_name, '') + args.dir_name + '_result.txt', 'a') as f:
        f.write(s + '\n')

    # 开始训练
    for epoch in range(args.num_epochs):
        # 训练的时候加上 model.train()
        model.train()
        model.cuda()
        exp_lr_scheduler.step()  # lr 衰减的函数

        lr = optimizer.param_groups[0]['lr']  # lr等于args.lr参数列表中给的学习率，主函数在调用train函数之前
        tq = tqdm.tqdm(total=len(dataloader_train)*args.batch_size)  # 进度条bar的个数 424/424（一个epoch图片个数）
        tq.set_description('epoch %d, lr %f' % (epoch+1, lr))        # 进度条前面信息的补充
        loss_record = []
        hist = np.zeros((args.num_classes, args.num_classes))
        evaluator = Evaluator(args.num_classes)  # epsilon用来防止0作为被除数

        for i, (img, label) in enumerate(dataloader_train):
            img, label = img.cuda(), label.cuda()

            # 在起步的时候减小学习率防止数据震荡（因为开始的时候参数是随机初始化的）
            if args.warmup == 1 and epoch == 0:
                lr = args.lr / (len(dataloader_train) - i)
                tq.set_description('epoch %d, lr %f' % (epoch + 1, lr))

                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            ######  单loss  #####
            output = model(img)  # [batch_size,n_cl,h,w]

            ######  多loss  #####
            # output,out1,out2,out3,out4,out5 = model(img)
            # output, out1, out2, out3, out4 = model(img)
            # output, out1, out2, out3 = model(img)
            # output, out1, out2 = model(img)
            # output, out1 = model(img)  # pspnet
            # print(output.shape)

            ###### 计算评价指标  #####
            predict = torch.argmax(output, 1)
            # 将训练结果放到cpu上去计算
            pre = predict.data.view(-1).cpu().numpy()
            lab = label.data.view(-1).cpu().numpy()
            hist += fast_hist(pre, lab, args.num_classes) # 计算hist混淆矩阵

            #########################################loss 方案########################################################
            # loss = criterion(output, label)
            loss = criterion(output, label)
            # loss1 = criterion(out1, label)
            # loss2 = criterion(out2, label)
            # loss3 = criterion(out3, label)
            # loss4 = criterion(out2, label)

            # loss6 = criterion(out5, label)
            # loss = loss + loss1 + loss2 + loss3
            # loss7 = loss2+loss3+loss4+loss5+loss6
            # loss = loss1+0.1*loss7
            # loss = loss1 + loss2 + loss3 + loss4 + loss5
            # loss = loss+loss1

            # loss = loss1 + 0.4*loss2 + 0.3*loss3 + 0.2*loss4 + 0.1*loss5
            # loss = loss + loss1 + loss2 + loss3 + loss4# pspnet

            tq.update(args.batch_size)    # 每次更新进度条的长度（每个bathsize跳1%）和下面的补充信息loss
            tq.set_postfix(loss='%.6f' % loss)  # 输入一个字典，显示实验指标（loss）

            optimizer.zero_grad()  # 在loss反向传播时，先进行梯度归零
            loss.backward()
            optimizer.step()

            # computer mean loss
            loss_record.append(loss.item())  # loss

        # computer miou
        iou_train_mean = evaluator.mean_intersection_over_union(hist)

        tq.close()
        # loss_record = torch.Tensor(loss_record)
        # loss_train_mean = torch.mean(loss_record)   #我这里改了一下，改成在tensor中计算平均,之后改回来试一下
        loss_train_mean = np.mean(loss_record)    # 发现cpu快

        print('Loss for train :{:.6f}'.format(loss_train_mean))
        print('Miou for train :{:.6f}'.format(iou_train_mean))

        # written to the log
        writer.add_scalar('{}_loss'.format('train'), loss_train_mean, epoch+1)   # 可视化工具里添加标量
        writer.add_scalar('{}_miou'.format('train'), iou_train_mean, epoch+1)   # 可视化工具里添加标量
        # 每隔args.checkpoint_step保存模型的参数字典
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            save_path = args.save_model_path + 'epoch_{:}'.format(epoch)
            torch.save(model.state_dict(), save_path)

        # 每个epoch记录验证集miou
        if epoch % args.validation_step == 0:
            miou = val(args,
                       model,
                       criterion,
                       args.num_classes,
                       csv_path,
                       dataloader_val,
                       epoch,
                       loss_train_mean,
                       iou_train_mean,
                       writer)

            # 如果验证集miou比之前高,保存模型的参数字典
            if miou > miou_max:
                save_path = args.summary_path+args.dir_name+'/checkpoints/'+'miou_{:.6f}.pth'.format(miou)
                torch.save(model.state_dict(), save_path)
                miou_max = miou

                # 记录最高的一次miou的模型参数字典
                miou_max_save_path = '{}{}/miou_{:.6f}_{:d}.pth'.format(args.summary_path, args.dir_name, miou_max,
                                                                        epoch+1)
                miou_max_save_model_dict = model.state_dict()

                predict_on_image(model,
                                 args.crop_height,
                                 args.crop_width,
                                 csv_path,
                                 read_path=args.summary_path+'25_10.png',
                                 save_path='{}{}/miou_{:.6f}_{:d}.png'
                                 .format(args.summary_path, args.dir_name, miou_max, epoch+1))  # _npy

    writer.close()
    save_path = args.save_model_path + 'last.pth'
    torch.save(miou_max_save_model_dict, miou_max_save_path)
    torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    # from torch.utils.data import DataLoader
    # from loaddata import Road_Seg
    #
    # transform = transforms.Compose([transforms.Resize(512),
    #                                 transforms.CenterCrop(512)])
    # data = Road_Seg('./data/train_512/images',
    #                 './data/train_512/labels',
    #                 './data/class_dict.csv',
    #                 transform = transform)
    # print(len(data))
    # dataloader_test = DataLoader(
    #     data,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=0
    # )
    #
    # model=torch.nn.Conv2d(3,2,kernel_size=1)
    #
    # with torch.no_grad():
    #     model.cuda()
    #     model.eval()
    #
    #     for i, (img, label) in enumerate(dataloader_test):
    #         img, label = img.cuda(), label.cuda()
    #
    #         predict = model(img)
    #         predict = predict.squeeze()  # [1, n_cl, 512, 512] ==> [n_cl, 512, 512]
    #
    #         predict = reverse_one_hot(predict)
    #
    #         label = label.squeeze()  # [1, 512, 512] ==> [512, 512]
    #
    #         break
    output=torch.randn(6,3,512,512)
    out=torch.argmax(output, 1)
    print(out.shape)
    print(out.view(-1))

