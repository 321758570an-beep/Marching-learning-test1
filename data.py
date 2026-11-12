import numpy as np
import pandas as pd
import glob
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_fscore_support

class_ids = list(range(1, 9))
class_names = {
    1: "N", 2: "L", 3: "R", 4: "V",
    5: "A", 6: "F", 7: "f", 8: "/"
}

# 读取现有划分文件：模型会用它们训练和评估
train = np.loadtxt("train/train_beats.csv", delimiter=",")
test = np.loadtxt("train/test_beats.csv", delimiter=",")

# 用 all_data.csv 重现原始划分，只用于核对测试集和计算 Train before
files = glob.glob("./*.csv")   # 只读 raw/ 下的原始文件
all_data = np.vstack([np.loadtxt(f, delimiter=",") for f in files])
raw_train, recreated_test = train_test_split(
    all_data,
    test_size=0.25,
    random_state=48
)

# 检查重现的测试集是否与现有 test_beats.csv 完全对应
if recreated_test.shape != test.shape or not np.allclose(
    recreated_test, test, atol=1e-5
):
    raise ValueError(
        "重现的测试集与 train/test_beats.csv 不一致。"
        "请确认 all_data.csv、划分比例和 random_state 与生成 CSV 时相同。"
    )

# 统计各类数量
y_raw_train = raw_train[:, -2].astype(int)
y_train = train[:, -2].astype(int)
y_test = test[:, -2].astype(int)

train_before = {
    c: int(np.sum(y_raw_train == c)) for c in class_ids
}
train_after = {
    c: int(np.sum(y_train == c)) for c in class_ids
}
test_counts = {
    c: int(np.sum(y_test == c)) for c in class_ids
}

# 模型输入只使用 275 个波形点，不包含类别标签和患者编号
X_train = train[:, :275]
X_test = test[:, :275]

svm_model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced"
    )
)

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# 计算逐类别测试指标
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    labels=class_ids,
    zero_division=0
)

rows = []
for i, c in enumerate(class_ids):
    if test_counts[c] == 0:
        p, r, f = "--", "--", "--"
    else:
        p = f"{precision[i]:.4f}"
        r = f"{recall[i]:.4f}"
        f = f"{f1[i]:.4f}"

    rows.append({
        "Class": class_names[c],
        "Train before": f"{train_before[c]:,}",
        "Train after": f"{train_after[c]:,}",
        "Test": f"{test_counts[c]:,}",
        "Precision": p,
        "Recall": r,
        "F1": f
    })

result_table = pd.DataFrame(rows)

print(result_table.to_string(index=False))
print("\nLaTeX table:")
print(result_table.to_latex(index=False, escape=False))