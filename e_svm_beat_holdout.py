import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 读取数据：前 275 列是心电波形，倒数第二列是类别，
# 最后一列是患者编号
train = np.loadtxt("train/train_beats.csv", delimiter=",")
test = np.loadtxt("train/test_beats.csv", delimiter=",")

# 排除类别 0（未确定类别），本实验评估类别 1–8
train = train[train[:, -2] != 0]
test = test[test[:, -2] != 0]

X_train = train[:, :275]
y_train = train[:, -2].astype(int)

X_test = test[:, :275]
y_test = test[:, -2].astype(int)

class_ids = list(range(1, 9))

print("训练集:", X_train.shape, y_train.shape)
print("测试集:", X_test.shape, y_test.shape)
print("训练集类别:", np.unique(y_train, return_counts=True))
print("测试集类别:", np.unique(y_test, return_counts=True))

# 建立并训练 SVM 分类器
svm_model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", class_weight="balanced")
)

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# 评估结果：所有分类指标统一按类别 1–8 计算
print("\nE1: SVM + Beat Hold-out")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision (macro): {precision_score(y_test, y_pred, labels=class_ids, average='macro', zero_division=0):.4f}")
print(f"Recall (macro):    {recall_score(y_test, y_pred, labels=class_ids, average='macro', zero_division=0):.4f}")
print(f"F1 Score (macro):  {f1_score(y_test, y_pred, labels=class_ids, average='macro', zero_division=0):.4f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred,
    labels=class_ids,
    zero_division=0
))

# 混淆矩阵：打印数值并绘图
cm = confusion_matrix(y_test, y_pred, labels=class_ids)
print("Confusion Matrix（行=真实类别，列=预测类别）:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_ids,
    yticklabels=class_ids
)
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("E1: SVM + Beat Hold-out")
plt.tight_layout()
plt.show()


# ---------------- PFI：置换特征重要性 ----------------

rng = np.random.default_rng(42)
n_repeats = 5

baseline_f1 = f1_score(
    y_test,
    y_pred,
    labels=class_ids,
    average="macro",
    zero_division=0
)

# 275 个采样点分成 11 个区域，每区 25 点
region_importance = np.zeros(11)

for region in range(11):
    start = region * 25
    end = start + 25
    score_drops = []

    for _ in range(n_repeats):
        X_permuted = X_test.copy()

        # 在不同心跳样本之间打乱当前区域，
        # 保留该区域内部 25 个采样点的相对结构
        shuffled_rows = rng.permutation(len(X_permuted))
        X_permuted[:, start:end] = X_test[shuffled_rows, start:end]

        permuted_pred = svm_model.predict(X_permuted)

        permuted_f1 = f1_score(
            y_test,
            permuted_pred,
            labels=class_ids,
            average="macro",
            zero_division=0
        )

        score_drops.append(baseline_f1 - permuted_f1)

    region_importance[region] = np.mean(score_drops)

print("\n各区域 PFI（macro-F1 下降量）:")
for region, importance in enumerate(region_importance, start=1):
    print(f"区域 {region}: {importance:.6f}")

plt.figure(figsize=(10, 4))
plt.bar(np.arange(1, 12), region_importance)
plt.xlabel("心跳时间区域（每区 25 个采样点）")
plt.ylabel("Macro-F1 下降量")
plt.title("E1: SVM Beat Hold-out PFI")
plt.xticks(np.arange(1, 12))
plt.tight_layout()
plt.show()