# train_model_v2.py —— Day 12 特征工程版
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 特征缩放（只用训练集 fit！）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # ⚠️ 用训练集的参数 transform 测试集！


# 4. 训练模型
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. 评估
acc = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"✅ 使用 StandardScaler 后准确率: {acc:.2%}")

# 6. 保存模型 + 缩放器（两者都要！）
joblib.dump(model, 'iris_model_v2_logistic_scaled.joblib')
joblib.dump(scaler, 'iris_scaler_v2.joblib')  # 👈 关键！预测时也要缩放
print("💾 模型和缩放器已保存")