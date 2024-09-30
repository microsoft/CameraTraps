__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import math
import copy
import sklearn.neighbors
import numpy

from sklearn.metrics import confusion_matrix
from itertools import tee
from typing import Optional, List

from ..data import BoundingBox
from .utils import bboxes_iou

from ..utils.registry import Registry

METRICS = Registry('metrics', module_key='animaloc.eval.metrics')

__all__ = ['METRICS', *METRICS.registry_names]

@METRICS.register()
class Metrics:
    '''
    Class to accumulate classification, detection and counting metrics, i.e.: 

        - Precision
        - Recall
        - F-beta score
        - Mean Absolute Error (MAE)
        - Mean Squared Error (MSE)
        - Root Mean Squared Error (RMSE)
        - Average Precision (AP)
        - Interclass confusion
        - Classification accuracy

    for binary or multiclass model.

    First, instanciate with a data matching threshold (e.g. IoU, radius),
    then feed the object with ground truth and predictions. You can retrieve 
    any metric at any time by calling the corresponding attribute.
    '''
    
    def __init__(self, threshold: float, num_classes: int = 2) -> None:
        '''
        Args:
            threshold (float): data matching threshold
            num_classes (int, optional): number of classes, background included. 
                Defaults to 2 (binary case).
        '''
        
        self.threshold = threshold
        self.num_classes = num_classes

        self.detections = []
        self.idx = 0

        self.tp = self._init_attr()
        self.fp = self._init_attr()
        self.fn = self._init_attr()
    
        self._sum_absolute_error = self._init_attr()
        self._sum_squared_error = self._init_attr()
        self._n_calls = self._init_attr()
        self._agg_sum_absolute_error = 0
        self._agg_sum_squared_error = 0
        self._total_calls = 0
        self._total_count = self._init_attr()

        self._ap_tables = self._init_attr(val=[])
        
        self.confusion_matrix = numpy.zeros((self.num_classes-1,self.num_classes-1))
        self._confusion_matrix = self.confusion_matrix

    def feed(self, gt: dict, preds: dict, est_count: Optional[list] = None) -> None:
        ''' Feed the object with ground truth and predictions and returns
        specified metrics optionally.

        Args:
            gt (dict): ground truth containing a dict with 'loc' and 'labels' 
                keys and list as values.
            preds (dict): predictions containing a dict with 'loc' and 'labels' 
                keys and list as values. Can contain an optional 'scores' key. 
            est_count (list, optional): list containing estimated count for each
                class, background excluded. Defaults to None.
        '''

        assert isinstance(est_count, (type(None), list))
        for o in [gt, preds]:
            assert isinstance(o, dict)
            assert len(o['loc']) == len(o['labels'])
        
        self.score_flag = 0
        if 'scores' in preds.keys():
            self.score_flag = 1
            assert len(preds['scores']) == len(preds['loc'])
        
        if len(gt['loc']) == 0:
            self._no_gt(gt, preds)
        
        if len(preds['loc']) == 0:
            self._no_preds(gt, preds)
        
        if len(gt['loc']) > 0 and len(preds['loc']) > 0:
            self.matching(gt, preds)
        
        if est_count is not None:
            gt_count = self._init_attr(0)
            if len(gt['loc']) > 0:
                gt_count = [gt['labels'].count(i+1) for i in range(self.num_classes-1)]

            self._update_errors(gt_count, est_count)
            self._update_calls(gt_count, est_count)
        
            self._total_calls += 1
            self._total_count = [self._total_count[i] + count for i, count in enumerate(est_count)]
        
        self._store_detections(preds, est_count)
        self.idx += 1
    
    def matching(self, gt: dict, preds: dict) -> None:
        ''' Method to match ground truth and predictions.
        To be overriden by subclasses
        
        Args:
            gt (dict): ground truth containing a dict with 'loc' and 'labels' 
                keys and list as values.
            preds (dict): predictions containing a dict with 'loc' and 'labels' 
                keys and list as values. Can contain an optional 'scores' key. 
        '''
        pass

    def copy(self):
        clone = copy.deepcopy(self)
        return clone
    
    def flush(self) -> None:
        ''' Flush the object '''

        self.detections = []
        self.idx = 0
        
        self.tp = self._init_attr()
        self.fp = self._init_attr()
        self.fn = self._init_attr()
    
        self._sum_absolute_error = self._init_attr()
        self._sum_squared_error = self._init_attr()
        self._n_calls = self._init_attr()
        self._agg_sum_absolute_error = 0
        self._agg_sum_squared_error = 0
        self._total_calls = 0
        self._total_count = self._init_attr()

        self._ap_tables = self._init_attr(val=[])

        self.confusion_matrix = numpy.zeros((self.num_classes-1,self.num_classes-1))
        self._confusion_matrix = self.confusion_matrix
    
    def aggregate(self) -> None:
        ''' Aggregate the metrics.

        By default, the classes are aggregated into a single class and the metrics are 
        therefore relative to the object vs. background configuration.
        '''

        inter = int(self._confusion_matrix.sum()) - sum(self.tp)
        
        self.fp = [sum(self.fp) - inter]
        self.fn = [sum(self.fn) - inter]
        self.tp = [int(self._confusion_matrix.sum())]
        self._sum_absolute_error = [self._agg_sum_absolute_error]
        self._sum_squared_error = [self._agg_sum_squared_error]
        self._n_calls = [self._total_calls]
        self._ap_tables = [[[1,*x[1:]] for x in sum(self._ap_tables, [])]]
        self._confusion_matrix = numpy.array([[1.]])
        self._total_count = [sum(self._total_count)]

    def precision(self, c: int = 1) -> float:
        ''' Precision 
        Args:
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
        '''

        c = c - 1
        if self.tp[c] > 0:
            return float(self.tp[c] / (self.tp[c] + self.fp[c]))
        else:
            return float(0)
    
    def recall(self, c: int = 1) -> float:
        ''' Recall 
        Args:
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
        '''
        
        c = c - 1
        if self.tp[c] > 0:
            return float(self.tp[c] / (self.tp[c] + self.fn[c]))
        else:
            return float(0)
    
    def fbeta_score(self, c: int = 1, beta: int = 1) -> float:
        ''' F-beta score 
        Args:
            c (int, optional): class id. Defaults to 1.
            beta (int, optional): beta value. Defaults to 1.
        
        Returns:
            float
        '''
        
        if self.tp[c-1] > 0:
            return float(
                (1 + beta**2)*self.precision(c)*self.recall(c) / 
                ((beta**2)*self.precision(c) + self.recall(c))
                )
        else:
            return float(0)        
    
    def mae(self, c: int = 1) -> float:
        ''' Mean Absolute Error 
        Args:
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
        '''

        c = c - 1
        return float(self._sum_absolute_error[c] / self._n_calls[c]) \
            if self._n_calls[c] else 0.
    
    def mse(self, c: int = 1) -> float:
        ''' Mean Squared Error 
        Args:
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
        '''
        c = c - 1
        return float(self._sum_squared_error[c] / self._n_calls[c]) \
            if self._n_calls[c] else 0.
    
    def rmse(self, c: int = 1) -> float:
        ''' Root Mean Squared Error 
        Args:
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
        '''
        return float(math.sqrt(self.mse(c)))
    
    def ap(self, c: int = 1) -> float:
        ''' Average Precision
        Args: 
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
        '''

        recalls, precisions = self.rec_pre_lists(c)
        
        if len(recalls) == 0 or len(precisions) == 0:
            return 0.
        else:
            return self._compute_AP(recalls, precisions)
    
    def rec_pre_lists(self, c: int = 1) -> tuple:
        ''' Recalls and Precisions lists
        Args: 
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            tuple
                recalls and precisions
        '''

        c = c - 1

        if len(self._ap_tables[c]) == 0:
            return [], []

        else:
            n_gt = self.fn[c] + self.tp[c]

            sorted_table = sorted(self._ap_tables[c], key=lambda x: x[1], reverse=True)
            sorted_table = numpy.array(sorted_table)
            sorted_table[:,2] = numpy.cumsum(sorted_table[:,2], axis=0)
            sorted_table[:,3] = numpy.cumsum(sorted_table[:,3], axis=0)

            precisions = sorted_table[:,2] / (sorted_table[:,2]+sorted_table[:,3])
            recalls = sorted_table[:,2] / n_gt

            return recalls.tolist(), precisions.tolist()
    
    def confusion(self, c: int = 1) -> float:
        ''' Interclass confusion
        Args: 
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
                interclass confusion
        '''
        c = c - 1
        cm_row = self._confusion_matrix[c]
        p = cm_row[c]/sum(cm_row) if sum(cm_row) else 0.
        return 1 - p
    
    def accuracy(self) -> float:
        ''' Classification accuracy 
        
        Returns:
            float
        '''

        N = self.confusion_matrix.sum()
        tp = self.confusion_matrix.diagonal().sum()
        if N > 0:
            return tp / N
        else:
            return 0.
    
    def total_count(self, c: int = 1) -> float:
        ''' Total class count
        Args: 
            c (int, optional): class id. Defaults to 1.
        
        Returns:
            float
                count
        '''
        c = c - 1
        return self._total_count[c]
        
    def _init_attr(self, val: int = 0) -> list:
        return [val] * (self.num_classes - 1)
    
    def _update_calls(self, gt_count: list, est_count: list):

        for i, _ in enumerate(self._n_calls):
            if gt_count[i] != 0 or est_count[i] != 0:
                self._n_calls[i] += 1

    def _update_errors(self, gt_count: list, est_count: list):

        for i, (count, est) in enumerate(zip(gt_count, est_count)):
            error = abs(count - est)
            squared_error = error**2

            self._sum_absolute_error[i] += error
            self._sum_squared_error[i] += squared_error
        
        agg_error = abs(sum(gt_count) - sum(est_count))
        agg_squared_error = agg_error**2
        self._agg_sum_absolute_error += agg_error
        self._agg_sum_squared_error += agg_squared_error
    
    def _no_gt(self, gt: dict, preds: dict) -> None:

        for c in range(1, self.num_classes):
            n_pred = len([lab for lab in preds['labels'] if lab == c])

            self.tp[c-1] += 0
            self.fp[c-1] += n_pred
            self.fn[c-1] += 0

            if self.score_flag:
                preds_fp = [[preds['labels'][i],preds['scores'][i],0,1]
                                  for i, _ in enumerate(preds['labels'])
                                  if preds['labels'][i] == c]

                self._ap_tables[c-1] = [*self._ap_tables[c-1], *preds_fp]
    
    def _no_preds(self, gt: dict, preds: dict) -> None:

        for c in range(1, self.num_classes):
            n_gt = len([lab for lab in gt['labels'] if lab == c])

            self.tp[c-1] += 0
            self.fp[c-1] += 0
            self.fn[c-1] += n_gt
    
    def _compute_AP(self, recalls: list, precisions: list) -> float:
        '''
        Compute the VOC Average Precision
        Code from: https://github.com/Cartucho/mAP
        (adapted from official matlab code VOC2012)
        '''

        recalls.insert(0, 0.0)
        recalls.append(1.0)
        precisions.insert(0, 0.0) 
        precisions.append(0.0) 

        mrec, mpre = recalls[:], precisions[:]

        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])

        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) 

        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        
        return ap
    
    def _store_detections(self, preds: dict, est_count: Optional[list] = None) -> None:
        ''' Store detections internally (1 row = 1 detection) '''

        m = map(dict, zip( * [
            [(k, v) for v in value]
            for k, value in preds.items()
            ]))
        m, m_copy = tee(m)
        
        counts = {}
        if est_count is not None:
            counts = {f'count_{i+1}': x for i, x in enumerate(est_count)}

        if len([x for x in m_copy]) > 0:
            for det in m:
                self.detections.append({'images': self.idx, **det, **counts})
        else:
            self.detections.append({'images': self.idx, **counts})
    
@METRICS.register()
class PointsMetrics(Metrics):
    ''' Metrics class for points (must be in (x,y) format) '''

    def __init__(self, radius: float, num_classes: int = 2) -> None:
        '''
        Args:
            radius (float): distance between ground truth and predicted point
                from which a point is characterizd as true positive
            num_classes (int, optional): number of classes, background included. 
                Defaults to 2 (binary case).
        '''
        super().__init__(threshold=radius, num_classes=num_classes)
    
    def matching(self, gt: dict, preds: dict) -> None:
        
        # matching
        dist = sklearn.neighbors.NearestNeighbors(n_neighbors=1, metric='euclidean').fit(preds['loc'])
        dist, idx = dist.kneighbors(gt['loc'])
        match_gt = [(k, d, i) for k, (d, i) in enumerate(zip(dist[:,0], idx[:,0]))]

        # sort according to distance
        match_gt = sorted(match_gt, key = lambda tup: tup[1])              
        # discard duplicates
        k_discard, i_discard = [], []
        filter_match_gt = []
        for k, d, i in match_gt:
            if k not in k_discard and i not in i_discard:
                filter_match_gt.append((k,d,i))
                k_discard.append(k), i_discard.append(i)
        # threshold
        filter_match_gt = [(k, d, i) for k, d, i in filter_match_gt if d <= self.threshold]

        # confusion matrix
        y_true = [gt['labels'][k] for k, d, i in filter_match_gt]
        y_pred = [preds['labels'][i] for k, d, i in filter_match_gt]

        self._confusion_matrix += confusion_matrix(
            y_true, y_pred, labels=list(range(1, self.num_classes)))

        for c in range(1, self.num_classes):
            n_gt = len([lab for lab in gt['labels'] if lab == c])
            n_pred = len([lab for lab in preds['labels'] if lab == c])

            lab_match = [(d, i) for k, d, i in filter_match_gt 
                            if gt['labels'][k] == preds['labels'][i] == c]

            tp = len(lab_match)
            self.tp[c-1] += tp
            self.fp[c-1] += (n_pred - tp)
            self.fn[c-1] += (n_gt - tp)

            if self.score_flag:
                tp_ids = [i for d, i in lab_match]
                preds_tp = [[preds['labels'][i],preds['scores'][i],1,0]
                              for _, i in lab_match]
                preds_fp = [[preds['labels'][i],preds['scores'][i],0,1]
                              for i, _ in enumerate(preds['labels'])
                              if preds['labels'][i] == c and i not in tp_ids]

                self._ap_tables[c-1] = [*self._ap_tables[c-1], *preds_tp, *preds_fp]
    
    def _store_detections(self, preds: dict, est_count: Optional[list] = None) -> None:

        m = map(dict, zip( * [
            [(k, v) for v in value]
            for k, value in preds.items()
            ]))
        m, m_copy = tee(m)
        
        counts = {}
        if est_count is not None:
            counts = {f'count_{i+1}': x for i, x in enumerate(est_count)}

        if len([x for x in m_copy]) > 0:
            for det in m:
                y, x = det['loc']
                det.update(dict(x=x, y=y))
                _ = det.pop('loc')
                self.detections.append({'images': self.idx, **det, **counts})
        else:
            self.detections.append({'images': self.idx, **counts})

@METRICS.register()
class BoxesMetrics(Metrics):
    ''' Metrics class for bounding boxes 
    (must be in (x_min, y_min, x_max, y_max) format '''

    def __init__(self, iou: float, num_classes: int = 2) -> None:
        '''
        Args:
            iou (float): Intersect-over-Union (IoU) threshold used to define a true
                positive.
            num_classes (int, optional): number of classes, background included. 
                Defaults to 2 (binary case).
        '''
        super().__init__(threshold=iou, num_classes=num_classes)
    
    def matching(self, gt: dict, preds: dict) -> None:

        ious, idx = self._most_overlapping_boxes(gt['loc'], preds['loc'])
        match_gt = [(k, iou, i) for k, (iou, i) in enumerate(zip(ious, idx)) 
                        if iou >= self.threshold]
        
        # confusion matrix
        y_true = [gt['labels'][k] for k, d, i in match_gt]
        y_pred = [preds['labels'][i] for k, d, i in match_gt]

        self._confusion_matrix += confusion_matrix(
            y_true, y_pred, labels=list(range(1, self.num_classes)))
        
        for c in range(1, self.num_classes):
            n_gt = len([lab for lab in gt['labels'] if lab == c])
            n_pred = len([lab for lab in preds['labels'] if lab == c])

            lab_match = [(d, i) for k, d, i in match_gt 
                            if gt['labels'][k] == preds['labels'][i] == c]

            tp = len(lab_match)
            self.tp[c-1] += tp
            self.fp[c-1] += (n_pred - tp)
            self.fn[c-1] += (n_gt - tp)

            if self.score_flag:
                tp_ids = [i for d, i in lab_match]
                preds_tp = [[preds['labels'][i],preds['scores'][i],1,0]
                              for _, i in lab_match]
                preds_fp = [[preds['labels'][i],preds['scores'][i],0,1]
                              for i, _ in enumerate(preds['labels'])
                              if preds['labels'][i] == c and i not in tp_ids]

                self._ap_tables[c-1] = [*self._ap_tables[c-1], *preds_tp, *preds_fp]
    
    def _most_overlapping_boxes(
        self, 
        gt_boxes: List[tuple], 
        preds_boxes: List[tuple], 
        ) -> tuple:
        
        gt_boxes = [BoundingBox(*coord) for coord in gt_boxes]
        preds_boxes = [BoundingBox(*coord) for coord in preds_boxes]

        iou_matrix = bboxes_iou(gt_boxes, preds_boxes)

        match_idx = []
        ious = []
        for row in iou_matrix:
            filt_row = [(k, elem) for k, elem in enumerate(row) if k not in match_idx]
            if len(filt_row) > 0:
                idx, iou_max = max(filt_row, key=lambda item:item[1])

                match_idx.append(idx)
                ious.append(iou_max)
        
        return ious, match_idx
    
    def _store_detections(self, preds: dict, est_count: Optional[list] = None) -> None:

        m = map(dict, zip( * [
            [(k, v) for v in value]
            for k, value in preds.items()
            ]))
        m, m_copy = tee(m)
        
        counts = {}
        if est_count is not None:
            counts = {f'count_{i+1}': x for i, x in enumerate(est_count)}

        if len([x for x in m_copy]) > 0:
            for det in m:
                x_min, y_min, x_max, y_max = det['loc']
                det.update(dict(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max))
                _ = det.pop('loc')
                self.detections.append({'images': self.idx, **det, **counts})
        else:
            self.detections.append({'images': self.idx, **counts})

@METRICS.register()
class ImageLevelMetrics(Metrics):
    ''' Metrics class for image-level classification '''

    def __init__(self, num_classes: int = 2) -> None:
        num_classes = num_classes + 1 # for convenience
        super().__init__(0, num_classes)
    
    def feed(self, gt: int, pred: int) -> tuple:
        '''
        Args:
            gt (int): numeric ground truth label
            pred (int): numeric predicted label
        '''

        gt = dict(labels=[gt], loc=[(0,0)])
        preds = dict(labels=[pred], loc=[(0,0)])
        
        super().feed(gt, preds)
    
    def matching(self, gt: dict, pred: dict) -> None:
        gt_lab = gt['labels'][0]
        p_lab = pred['labels'][0]

        if gt_lab == p_lab:
            self.tp[gt_lab-1] += 1
        else:
            self.fp[p_lab-1] += 1
            self.fn[gt_lab-1] += 1
        
        self._confusion_matrix += confusion_matrix(
            [gt_lab], [p_lab], labels=list(range(self.num_classes-1)))