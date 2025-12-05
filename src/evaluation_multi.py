import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    top_k_accuracy_score, precision_recall_fscore_support, log_loss
)

COLORS = {
    'hightlight_blue': '#4c72b0',      # Deep Blue (Best Accuracy)
    'hightlight_red': '#c44e52',       # Deep Red (FPR Bar)
    'light_gray': '#e1e1e1',     # Light Gray
    'darker_gray': '#a0a0a0',    # Darker Gray (Other FPRs)
    'text_color': '#333333'
}

def show_confusion_matrix(y, y_pred, class_names, model=None, do_save=False):
    print("1. Confusion Matrix\n")
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(max(12, len(class_names)/3), max(9, len(class_names)/4)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.title('Confusion Matrix (counts)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout() 
    plt.show()
    if do_save:
        plt.savefig(f'close_confusion_matrix_{model}.png')
    return cm

def per_class_accuracy(cm, class_names, model=None, do_save=False):
    print("2. Per-class Accuracy\n")
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    colors = []
    for acc in per_class_acc:
        if acc >= 0.95:
            colors.append(COLORS['hightlight_blue'])
        elif acc < 0.7:
            colors.append(COLORS['hightlight_red'])
        else:
            colors.append(COLORS['light_gray'])
    plt.figure(figsize=(max(12, len(class_names)/3), max(9, len(class_names)/4)))
    sns.barplot(x=class_names, y=per_class_acc, palette=colors)
    plt.ylim(0.4, 1.0)
    plt.title("Per-class Accuracy", color=COLORS['text_color'])
    plt.ylabel("Accuracy", color=COLORS['text_color'])
    for i, v in enumerate(per_class_acc):
        plt.text(i, v + 0.005, f"{v:.2f}", ha='center', color=COLORS['text_color'], fontsize=9)
    plt.tight_layout(); plt.show()
    if do_save:
        plt.savefig(f'close_per_class_accuracy_{model}.png')

def overall_metrics(y, y_pred, n_classes=None, y_prob=None, do_save=False):
    print("3. Overall Metrics\n")
    acc = accuracy_score(y, y_pred)

    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y, y_pred, average='macro', zero_division=0
    )

    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y, y_pred, average='weighted', zero_division=0
    )

    print(f"- Accuracy: {acc:.4f}")
    print(f"- Macro F1: {f1_m:.4f} (macro P={prec_m:.4f}, macro R={rec_m:.4f})")
    print(f"- Weighted F1: {f1_w:.4f} (weighted P={prec_w:.4f}, weighted R={rec_w:.4f})")

    # log loss
    try:
        ll = log_loss(y, y_prob, labels=list(range(n_classes)))
        print(f"- Log Loss: {ll:.4f}")
    except Exception as e:
        print(f"- Log Loss: N/A ({e})")

def top_k_accuracy(y, y_prob, class_names, do_save=False):
    print("4. Top k Accuracy\n")
    for k in [1, 3, 5]:
        k_eff = min(k, len(class_names))  # incase k > #classes
        topk_acc = top_k_accuracy_score(y, y_prob, k=k_eff, labels=class_names)
        print(f"- Top-{k_eff} accuracy: {topk_acc:.4f}")

def confidence_analysis(y, y_pred, y_prob, model=None, do_save=False):
    print("5. Confidence Analysis\n")
    max_conf = y_prob.max(axis=1)
    correct = (y_pred == y).astype(int)
    df_conf = pd.DataFrame({
        "confidence": max_conf,
        "correct": correct,
        "true_label": y,
        "pred_label": y_pred
    })

    plt.figure(figsize=(6,4))
    sns.histplot(data=df_conf, x="confidence", hue="correct", bins=10,
                palette={1: "seagreen", 0: "salmon"}, alpha=0.6, element="step", stat="density")
    plt.title("Confidence Distribution (green=correct, red=wrong)")
    plt.xlabel("max predicted probability")
    plt.tight_layout()
    plt.show()
    if do_save:
        plt.savefig(f'close_confidence_analysis_{model}.png')