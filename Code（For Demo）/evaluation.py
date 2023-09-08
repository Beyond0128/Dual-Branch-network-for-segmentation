import torch
import numpy as np
# 计算miou
# SR: Segmentation Result
# GT: Ground Truth
# TP（True Positive）：真正例，模型预测为正例，实际是正例（模型预测为类别1，实际是类别1）
# FP（False Positive）：假正例，模型预测为正例，实际是反例 （模型预测为类别1，实际是类别2）
# FN（False Negative）：假反例，模型预测为反例，实际是正例 （模型预测为类别2，实际是类别1）
# TN（True Negative）：真反例，模型预测为反例，实际是反例 （模型预测为类别2，实际是类别2）

# 准确率（Accuracy），对应：语义分割的像素准确率 PA
# 公式：Accuracy = (TP + TN) / (TP + TN + FP + FN)
# 意义：对角线计算。预测结果中正确的占总预测值的比例（对角线元素值的和 / 总元素值的和


# 计算混淆矩阵
def fast_hist(label_true, label_pred, n_class):
    # mask在和label_true相对应的索引的位置上填入true或者false
    # label_true[mask]会把mask中索引为true的元素输出
    mask = (label_true >= 0) & (label_true < n_class)
    # np.bincount()会给出索引对应的元素个数
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.epsilon = np.finfo(np.float32).eps  # 防止÷0变成nan

    def pixel_accuracy(self, hist):
        pa=np.diag(hist).sum() / hist.sum()
        return pa

    def mean_pixel_accuracy(self, hist):
        cpa = (np.diag(hist) + self.epsilon) / (hist.sum(axis=0) + self.epsilon)
        mpa = np.nanmean(cpa)  # nanmean会自动忽略nan的元素求平均,当epsilon为0时生效
        return mpa

    def precision(self, hist):
        precision = (np.diag(hist) + self.epsilon) / (hist.sum(axis=0) + self.epsilon)
        precision = np.nanmean(precision)
        return precision

    def recall(self, hist):
        recall = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + self.epsilon)
        recall = np.nanmean(recall)
        return recall

    def f1_score(self, hist):
        f1 = (np.diag(hist) + self.epsilon) * 2 / (hist.sum(axis=1) * 2 + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        f1 = np.nanmean(f1)
        return f1

    def mean_intersection_over_union(self, hist):
        iou = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        miou = np.nanmean(iou)
        return miou

    def frequency_weighted_intersection_over_union(self, hist):
        freq = hist.sum(axis=1) / hist.sum()
        iou = (np.diag(hist) + self.epsilon) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + self.epsilon)
        fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
        return fwiou


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    label = np.array(Image.open(r'D:\Desktop\ZDS\ZDS\L7_code\demo\0000_0_4.png').convert("P"))
    predict = np.array(Image.open(r'D:\Desktop\ZDS\ZDS\L7_code\demo\epoch_18.png').convert("P"))
    evaluator = Evaluator(3)  # epsilon用来防止0作为被除数
    hist = np.zeros((3, 3))
    hist += fast_hist(predict.flatten(), label.flatten(), 3)
    miou = evaluator.mean_intersection_over_union(hist)
    print(miou)