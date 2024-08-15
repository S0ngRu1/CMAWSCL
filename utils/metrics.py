
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score


def collect_metrics(dataset, y_true, y_pred):

    acc = accuracy_score(y_true, y_pred.argmax(1))
    f1 = f1_score(y_true, y_pred.argmax(1))
    auc = roc_auc_score(y_true, y_pred[:, 1])
    recall = recall_score(y_true, y_pred.argmax(1))


    eval_results = {
        "acc": round(acc, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4)
    }

    return eval_results

