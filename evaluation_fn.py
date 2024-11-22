from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def save_results(y_test, y_pred, filename):
    acc, precision, recall, f1 = get_metrics(y_test, y_pred)
    
    # save the results
    with open(filename, 'w') as f:
        f.write(f"Accuracy: {acc}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1: {f1}\n")

    # save the roc curve
    save_roc_curve(y_test, y_pred, filename.replace('.txt', '_roc.png'))
    # save the precision recall curve
    save_precision_recall_curve(y_test, y_pred, filename.replace('.txt', '_prc.png'))
    # save the confusion matrix
    save_confusion_matrix(y_test, y_pred, filename.replace('.txt', '_cm.png'))
    return acc, precision, recall, f1

def save_roc_curve(y_test, y_pred, filename):
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)
    
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f"Class {class_name} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title("Multi-Class ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

def save_precision_recall_curve(y_test, y_pred, filename):
    classes = np.unique(y_test)
    y_test_bin = label_binarize(y_test, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)
    
    precision = {}
    recall = {}
    for i in range(len(classes)):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_bin[:, i])

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(classes):
        plt.plot(recall[i], precision[i], lw=2, label=f"Class {class_name}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multi-Class Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.savefig(filename)
    plt.close()

def save_confusion_matrix(y_test, y_pred, filename):
    cm = confusion_matrix(y_test, y_pred)
    class_names = np.unique(y_test)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(filename)
    plt.close()

def convert_to_float(acc, precision, recall, f1):
    if not isinstance(acc, float):
        acc = acc.item() if isinstance(acc, np.ndarray) else acc[0]
    if not isinstance(precision, float):
        precision = precision.item() if isinstance(precision, np.ndarray) else precision[0]
    if not isinstance(recall, float):
        recall = recall.item() if isinstance(recall, np.ndarray) else recall[0]
    if not isinstance(f1, float):
        f1 = f1.item() if isinstance(f1, np.ndarray) else f1[0]
    return acc, precision, recall, f1

def get_metrics(actual: list, pred: list):
    acc = accuracy_score(actual, pred),
    precision = precision_score(actual, pred, average='macro', zero_division=0),
    recall = recall_score(actual, pred, average='macro', zero_division=0),
    f1 = f1_score(actual, pred, average='macro', zero_division=0)
    return convert_to_float(acc, precision, recall, f1)