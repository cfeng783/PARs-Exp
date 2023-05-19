import numpy as np
from sklearn.metrics import confusion_matrix



def search_best_f1(scores, labels, start, end=None, step_num=1, verbose=False):
    """
    Find the best standard f1 score by searching best `threshold` in [`start`, `end`).

    Parameters
    ----------
    scores : ndarray or list
        The anomaly scores of samples
    labels : ndarray or list
        The ground truth labels of samples
    start : float
        The start value for search
    end : float, default is None
        The end value for search, if None, then end=start
    step_num : int, default is 1
        The number of steps for search
    verbose: bool, default is True
        whether to display results whiling searching
        
    Returns
    -------
    list 
        list for results, [f1,precision,recall,TP,TN,FP,FN]
    float
        the selected anomaly threshold
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    best_metrics = [0]
    best_threshold = 0
    for _ in range(search_step):
        threshold += search_range / float(search_step)
        target = _calc_f1(scores, labels, threshold)
        if target[0] > best_metrics[0]:
            best_metrics = target
            best_threshold = threshold
    return best_metrics,best_threshold

def _calc_f1(score, label, threshold):
    """
    calculate f1 score by predict and actual.TP, TN, FP, FN

    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    predict = np.array(score)
    predict = predict > threshold

    TN,FP,FN,TP = confusion_matrix(label,predict).ravel()
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1,precision,recall,TP, TN, FP, FN



def calc_detection_performance(y_true, y_pred):
    """
    calculate anomaly detection performance

    Parameters
    ----------
    y_true : ndarray or list
        The ground truth labels
    y_pred : ndarray or list
        The predicted labels
    
    Returns
    -------
    list 
        list for results, [f1,precision,recall,TP,TN,FP,FN]
    """

    TN,FP,FN,TP = confusion_matrix(y_true,y_pred).ravel()
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return [f1,precision,recall,TP, TN, FP, FN]


def hitRateAtP(gts,pts,prob):
    l = round(len(gts)*prob)
    l = min(l,len(pts))
    hit = 0
    for i in range(l):
        ps = pts[i]
        if ps in gts:
            hit += 1
    return hit/len(gts)

def hitRates(gts,pts):
    hit100 = hitRateAtP(gts,pts,prob=1)
    hit150 = hitRateAtP(gts,pts,prob=1.5)
    return [hit100,hit150]
    