import numpy as np
import pandas as pd

# Get confusion matrix with rows as actual classes and columns as predicted classes
def get_confusion_matrix(predicted, actual):
  classes = np.unique(actual)
  n = len(classes)
  confusion_matrix = np.zeros((n,n))
  for a,p in zip(actual,predicted):
    confusion_matrix[a,p] += 1
  return confusion_matrix

# Get macro-averaged precision over all classes
def precision(confusion_matrix):
  avg_precision = 0
  for c in range(confusion_matrix.shape[0]):
    tp = confusion_matrix[c,c]
    fp = np.sum(confusion_matrix[:,c]) - tp
    avg_precision += tp / (tp + fp)
  return avg_precision / confusion_matrix.shape[0]

# Get macro-averaged recall over all classes
def recall(confusion_matrix):
  avg_recall = 0
  for c in range(confusion_matrix.shape[0]):
    tp = confusion_matrix[c,c]
    fn = np.sum(confusion_matrix[c,:]) - tp
    avg_recall += tp / (tp + fn)
  return avg_recall / confusion_matrix.shape[0]

# Get macro-averaged f1 score over all classes
def f1_score(confusion_matrix):
  avg_f1 = 0
  for c in range(confusion_matrix.shape[0]):
    tp = confusion_matrix[c,c]
    fp = np.sum(confusion_matrix[:,c]) - tp
    fn = np.sum(confusion_matrix[c,:]) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    avg_f1 += 2 * (precision * recall) / (precision + recall)
  return avg_f1 / confusion_matrix.shape[0]