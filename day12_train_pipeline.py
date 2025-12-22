# train_pipeline.py —— 更优雅的写法
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 创建 Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=200, random_state=42))
])

# 训练（自动先 scaler.fit，再 classifier.fit）
pipeline.fit(X_train, y_train)

# 评估
acc = accuracy_score(y_test, pipeline.predict(X_test))
print(f"✅ Pipeline 准确率: {acc:.2%}")

# 只需保存一个文件！
joblib.dump(pipeline, 'iris_pipeline_v2.joblib')
print("💾 Pipeline 已保存为 iris_pipeline_v2.joblib")