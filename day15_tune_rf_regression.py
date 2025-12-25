# tune_rf_regression.py —— Day 15 超参数调优（支持离线 & 与 Day13/14 一致）
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import numpy as np

# ==============================
# 1. 加载数据（与 Day 13/14 完全一致的 fallback 逻辑！）
# ==============================
print("📥 加载加州房价数据...")

try:
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    print("✅ 使用真实加州房价数据")
except Exception as e:
    print(f"⚠️ 真实数据加载失败 ({type(e).__name__})，切换到模拟数据...")
    from sklearn.datasets import make_regression
    
    # ⚠️ 关键：必须和 Day13/14 的模拟逻辑完全一致！
    X, y = make_regression(
        n_samples=20640,
        n_features=8,
        n_informative=6,
        noise=100,
        random_state=42
    )
    # 缩放到真实房价范围 [0.15, 5.0]（单位：千美元）
    y = (y - y.min()) / (y.max() - y.min())  # 归一化到 [0, 1]
    y = y * (5.0 - 0.15) + 0.15              # 缩放到 [0.15, 5.0]
    print("✅ 使用模拟加州房价数据（离线模式，与 Day13/14 一致）")

print(f"📊 数据规模: {X.shape[0]} 样本, {X.shape[1]} 特征")

# ==============================
# 2. 划分数据集（random_state=42 保证一致性）
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 3. 超参数调优
# ==============================
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],  # 新增：防止过拟合
    # 注意：random_state 不在此处搜索，直接在模型中固定
}

# 创建基础模型（固定 random_state 保证可复现）
rf = RandomForestRegressor(random_state=42)

print("🔍 开始超参数随机搜索（5 折交叉验证，20 组组合）...")
search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1,
    random_state=42  # 控制搜索的随机性
)

search.fit(X_train, y_train)

# ==============================
# 4. 输出结果 & 保存模型
# ==============================
print("\n✅ 最佳超参数:")
best_params = search.best_params_
for k, v in best_params.items():
    print(f"  {k}: {v}")

print(f"🏆 交叉验证最佳 R²: {search.best_score_:.4f}")

# 评估测试集
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print(f"\n🧪 测试集性能:")
print(f"  MAE: ${test_mae:.2f}k")
print(f"  R²: {test_r2:.4f}")

# 保存模型
os.makedirs('models', exist_ok=True)
model_path = 'E:/AI_learning/models/regressor_v2_rf_tuned.joblib'
joblib.dump(best_model, model_path)
print(f"\n💾 调优后模型已保存至: {model_path}")

# 在 tune_rf_regression.py 末尾添加
#  在 evals/ 目录下保存调优日志
import json
tuning_log = {
    'best_params': best_params,
    'cv_best_score': float(search.best_score_),
    'test_mae': float(test_mae),
    'test_r2': float(test_r2),
    'data_source': 'simulated (offline fallback)'
}
with open('E:/AI_learning/evals/tuning_log_day15.json', 'w') as f:
    json.dump(tuning_log, f, indent=2)