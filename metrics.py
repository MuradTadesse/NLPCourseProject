# src/evaluation/metrics.py
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class EvaluationMetrics:
    @staticmethod
    def compute_metrics(pred_labels, true_labels):
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted'
        )
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    @staticmethod
    def compute_confusion_matrix(pred_labels, true_labels, label_names):
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(true_labels, pred_labels)