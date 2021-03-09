# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2021-3-9
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2021 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import abc
import numpy as np


class VOC12mAP(object):
    def __init__(self, num_classes):
        super(VOC12mAP, self).__init__()
        self._num_classes = num_classes

    def reset(self):
        self._predicted = np.array([], dtype=np.float32).reshape(0, self._num_classes)
        self._gt_label = np.array([], dtype=np.float32).reshape(0, self._num_classes)

    def update(self, predicted, gt_label):
        self._predicted = np.vstack((self._predicted, predicted))
        self._gt_label = np.vstack((self._gt_label, gt_label))

    def compute(self):
        return self._voc12_mAP()
    
    def _voc12_mAP(self):
        sample_num, num_classes = self._gt_label.shape
        ap_list = []

        for class_id in range(num_classes):
            confidence = self._predicted[:, class_id]
            sorted_ind = np.argsort(-confidence)
            sorted_label = self._gt_label[sorted_ind, class_id]

            tp = (sorted_label == 1).astype(np.int64)   # true positive
            fp = (sorted_label == 0).astype(np.int64)   # false positive
            tp_num = max(sum(tp), np.finfo(np.float64).eps)
            tp = np.cumsum(tp)
            fp = np.cumsum(fp)
            recall = tp / float(tp_num)
            precision = tp / np.arange(1, sample_num + 1, dtype=np.float64)

            ap = self._voc_AP(recall, precision, tp_num)    # average precision
            ap_list.append(ap)

        mAP = np.mean(ap_list)  # mean average precision
        return ap_list, mAP

    def _voc_AP(self, recall, precision, tp_num):
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


class AverageLoss(object):
    def __init__(self):
        super(AverageLoss, self).__init__()

    def reset(self):
        self._sum = 0
        self._counter = 0

    def update(self, loss, n=0):
        self._sum += loss * n
        self._counter += n

    def compute(self):
        return self._sum / self._counter


class AverageMeter(object):
    def __init__(self, num_classes):
        super(AverageMeter, self).__init__()
        self.num_classes = num_classes

    def reset(self):
        self._right_pred_counter = np.zeros(self.num_classes)  # right predicted image per-class counter
        self._pred_counter = np.zeros(self.num_classes)    # predicted image per-class counter
        self._gt_counter = np.zeros(self.num_classes)  # ground-truth image per-class counter

    def update(self, confidence, gt_label):
        self._count(confidence, gt_label)

    def compute(self):
        self._op = sum(self._right_pred_counter) / sum(self._pred_counter)
        self._or = sum(self._right_pred_counter) / sum(self._gt_counter)
        self._of1 = 2 * self._op * self._or / (self._op + self._or)
        self._right_pred_counter = np.maximum(self._right_pred_counter, np.finfo(np.float64).eps)
        self._pred_counter = np.maximum(self._pred_counter, np.finfo(np.float64).eps)
        self._gt_counter = np.maximum(self._gt_counter, np.finfo(np.float64).eps)
        self._cp = np.mean(self._right_pred_counter / self._pred_counter)
        self._cr = np.mean(self._right_pred_counter / self._gt_counter)
        self._cf1 = 2 * self._cp * self._cr / (self._cp + self._cr)

    @abc.abstractmethod
    def _count(self, confidence, gt_label):
        pass

    @property
    def op(self):   # overall precision
        return self._op

    @property   # overall recall
    def or_(self):
        return self._or

    @property   # overall F1
    def of1(self):
        return self._of1

    @property   # per-class precision
    def cp(self):
        return self._cp

    @property   # per-class recall
    def cr(self):
        return self._cr

    @property   # per-class F1
    def cf1(self):
        return self._cf1


class TopkAverageMeter(AverageMeter):
    def __init__(self, num_classes, topk=3):
        super(TopkAverageMeter, self).__init__(num_classes)
        self.topk = topk

    def _count(self, confidence, gt_label):
        sample_num = confidence.shape[0]
        sorted_inds = np.argsort(-confidence, axis=-1)
        for i in range(sample_num):
            sample_gt_label = gt_label[i]
            topk_inds = sorted_inds[i][:self.topk]
            self._gt_counter[sample_gt_label == 1] += 1
            self._pred_counter[topk_inds] += 1
            correct_inds = topk_inds[sample_gt_label[topk_inds] == 1]
            self._right_pred_counter[correct_inds] += 1


class ThresholdAverageMeter(AverageMeter):
    def __init__(self, num_classes, threshold=0.5):
        super(ThresholdAverageMeter, self).__init__(num_classes)
        self.threshold = threshold

    def _count(self, confidence, gt_label):
        sample_num = confidence.shape[0]
        for i in range(sample_num):
            sample_gt_label = gt_label[i]
            self._gt_counter[sample_gt_label == 1] += 1
            inds = np.argwhere(confidence[i] > self.threshold).squeeze(-1)
            self._pred_counter[inds] += 1
            correct_inds = inds[sample_gt_label[inds] == 1]
            self._right_pred_counter[correct_inds] += 1
