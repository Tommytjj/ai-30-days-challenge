# train_regression.py —— Day 13 回归实战（支持离线 fallback）
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ==============================
# 1. 加载加州房价数据（带 fallback）
# ==============================
print("📥 正在加载加州房价数据...")

# 尝试加载真实数据
try:
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    print("✅ 成功加载真实加州房价数据！")
except Exception as e:
    print(f"⚠️ 无法加载真实数据 ({type(e).__name__}: {e})，正在切换到模拟数据...")
    
    # --- 使用模拟数据替代 ---
    from sklearn.datasets import make_regression
    
    X_sim, y_sim = make_regression(
        n_samples=20640,
        n_features=8,
        n_informative=6,
        noise=100,
        random_state=42
    )
    
    # 缩放目标值到合理房价范围 [0.15k, 5.0k]
    y_sim = (y_sim - y_sim.min()) / (y_sim.max() - y_sim.min())
    y_sim = y_sim * (5.0 - 0.15) + 0.15
    
    # 构造兼容对象
    class MockHousing:
        def __init__(self):
            self.data = X_sim
            self.target = y_sim
            self.feature_names = [
                'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude'
            ]
            self.DESCR = "Simulated California Housing Dataset (offline fallback)"
    
    housing = MockHousing()
    print("✅ 已切换到模拟加州房价数据（离线模式）")

# 提取特征和目标
X, y = housing.data, housing.target

print(f"📊 数据规模: {X.shape[0]} 条样本, {X.shape[1]} 个特征")
print(f"🏷️  特征名: {housing.feature_names}")
print(f"💰 目标（房价中位数）范围: ${y.min():.1f}k - ${y.max():.1f}k")

# ==============================
# 2~7. 原有训练流程（完全不变）
# ==============================

# 2. 划分训练集/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 构建回归 Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
     ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 4. 训练
pipeline.fit(X_train, y_train)

# 5. 预测
y_pred = pipeline.predict(X_test)

# 6. 评估（回归三大指标）
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n✅ 回归模型评估结果:")
print(f"  MAE (平均绝对误差): ${mae:.2f}k")
print(f"  RMSE (均方根误差): ${rmse:.2f}k")
print(f"  R² (决定系数): {r2:.4f} (越接近1越好)")

# 7. 保存模型
joblib.dump(pipeline, 'california_housing_pipeline_v1_rf.joblib')
print("\n💾 回归 Pipeline 已保存为 california_housing_pipeline_v1_rf.joblib")