import numpy as np
import torch
from torch import nn
# 计算混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    # mask在和label_true相对应的索引的位置上填入true或者false
    # label_true[mask]会把mask中索引为true的元素输出
    mask = (label_true >= 0) & (label_true < n_class)
    # np.bincount()会给出索引对应的元素个数
    """
    hist是一个混淆矩阵
    hist是一个二维数组，可以写成hist[label_true][label_pred]的形式
    最后得到的这个数组的意义就是行下标表示的类别预测成列下标类别的数量
    比如hist[0][1]就表示类别为1的像素点被预测成类别为0的数量
    对角线上就是预测正确的像素点个数
    n_class * label_true[mask].astype(int) + label_pred[mask]计算得到的是二维数组元素
    变成一位数组元素的时候的地址取值(每个元素大小为1)，返回的是一个numpy的list，然后
    np.bincount就可以计算各中取值的个数
    """
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist

"""
label_trues 正确的标签值
label_preds 模型输出的标签值
n_class 数据集中的分类数
"""
def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    # 一个batch里面可能有多个数据
    # 通过迭代器将一个个数据进行计算
    for lt, lp in zip(label_trues, label_preds):
        # numpy.ndarray.flatten将numpy对象拉成1维
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    
    # np.diag(a)假如a是一个二维矩阵，那么会输出矩阵的对角线元素
    # np.sum()可以计算出所有元素的和。如果axis=1，则表示按行相加
    """
    acc是准确率 = 预测正确的像素点个数/总的像素点个数
    acc_cls是预测的每一类别的准确率(比如第0行是预测的类别为0的准确率)，然后求平均
    iu是召回率Recall，公式上面给出了
    mean_iu就是对iu求了一个平均
    freq是每一类被预测到的频率
    fwavacc是频率乘以召回率，我也不知道这个指标代表什么
    """
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    # nanmean会自动忽略nan的元素求平均
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc



def get_weights(target):
    t_np = target.view(-1).data.cpu().numpy()

    classes, counts = np.unique(t_np, return_counts=True)
    cls_w = np.median(counts) / counts
    #cls_w = class_weight.compute_class_weight('balanced', classes, t_np)

    weights = np.ones(32)
    weights[classes] = cls_w
    return torch.from_numpy(weights).float().cuda()

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE =  nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        loss = self.CE(output, target)
        return loss
