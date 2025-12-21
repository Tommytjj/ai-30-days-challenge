# train_model.py —— 修复版
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 2. 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 定义模型
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=200, random_state=42)
}

# 4. 训练 + 评估 + 保存
trained_models = {}
for name, model in models.items():
    # 训练
    model.fit(X_train, y_train)
    
    # 评估
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:20} 准确率: {acc:.2%}")
    
    # 保存到字典（用于后续保存）
    trained_models[name] = model

# 5. 保存训练好的模型（带清晰命名）
joblib.dump(trained_models["LogisticRegression"], 'iris_model_v1_logistic.joblib')
joblib.dump(trained_models["SVM"], 'iris_model_v2_svm.joblib')
joblib.dump(trained_models["RandomForest"], 'iris_model_v3_randomforest.joblib')

print("\n✅ 所有模型已保存！")