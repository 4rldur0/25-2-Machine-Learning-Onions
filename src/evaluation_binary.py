import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

def show_confusion_matrix(y, y_pred, model, do_save=False):
    print("1. Confusion Matrix\n")

    print(classification_report(y, y_pred, target_names=['unmonitored', 'monitored']))

    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['unmonitored', 'monitored'],
                yticklabels=['unmonitored', 'monitored'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Prediction')
    if do_save:
        plt.savefig(f'open_confusion_matrix_{model}.png')

def show_roc_curve(y, y_prob, model, do_save=False):
    print("2. ROC Curve\n")

    fpr, tpr, thresholds_roc = roc_curve(y, y_prob)
    roc_auc = roc_auc_score(y, y_prob)
    print(f"- ROC AUC Score: {roc_auc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='black', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', alpha=0.7, label='Random Guess (AUC = 0.5)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity (False Positive Rate)')
    plt.ylabel('Sensitivity (True Positive Rate)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    if do_save:
        plt.savefig(f'open_roc_curve_{model}.png')

def show_pr_curve(y, y_prob, model, do_save=False):
    print("3. PR Curve\n")

    precision, recall, thresholds_pr = precision_recall_curve(y, y_prob)
    avg_precision = average_precision_score(y, y_prob)
    print(f"- Average Precision (AP) Score: {avg_precision:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='black', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
    random_guess = len(y[y==1]) / len(y)
    plt.plot([0, 1], [random_guess, random_guess], linestyle='--', color='gray', alpha=0.7, label=f'Random Guess = {random_guess:.2f}')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision (PPV)')
    plt.title('PR Curve')
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    if do_save:
        plt.savefig(f'open_pr_curve_{model}.png')

def show_prediction_score(y, y_prob, model, do_save=False):
    print("4. Prediction Score")

    df = pd.DataFrame({
        'prediction_score': y_prob,
        'class_label': y
    })
    threshold = 0.5
    plt.figure(figsize=(5, 6))
    sns.stripplot(x=[0]*len(df), y='prediction_score', hue='class_label', data=df,
                palette={0: 'blue', 1: 'orange'}, jitter=0.4, alpha=0.5, dodge=True,
                size=4, legend=False)
    plt.xticks([])
    plt.ylim(-2.0, 2.0)
    plt.ylabel('prediction score')
    plt.axhline(y=threshold, color='black', linestyle='-', linewidth=1.5, label='Threshold (0.5)')
    plt.scatter([], [], color='blue', label=f'unmonitored ({list(y).count(0)} samples)')
    plt.scatter([], [], color='orange', label=f'monitored ({list(y).count(1)} samples)')
    plt.legend(loc='upper right')
    plt.title('Prediction Score Distribution by Class')
    plt.tight_layout()
    if do_save:
        plt.savefig(f'prediction_score_distribution_{model}.png')