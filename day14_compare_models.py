# compare_models.py —— Day 14 模型对比（支持离线 fallback）
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

# ==============================
# 1. 加载数据（带 fallback，与训练时完全一致！）
# ==============================
print("📥 正在加载加州房价数据（用于评估）...")

try:
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    print("✅ 使用真实加州房价数据")
except Exception as e:
    print(f"⚠️ 真实数据加载失败 ({type(e).__name__})，切换到模拟数据...")
    from sklearn.datasets import make_regression
    
    X_sim, y_sim = make_regression(
        n_samples=20640,
        n_features=8,
        noise=100,
        random_state=42
    )
    # 缩放到合理房价范围 [0.15, 5.0]
    y_sim = (y_sim - y_sim.min()) / (y_sim.max() - y_sim.min())
    y_sim = y_sim * (5.0 - 0.15) + 0.15
    
    class MockHousing:
        def __init__(self):
            self.data = X_sim
            self.target = y_sim
            self.feature_names = [
                'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                'Population', 'AveOccup', 'Latitude', 'Longitude'
            ]
    housing = MockHousing()
    print("✅ 使用模拟加州房价数据（离线模式）")

X, y = housing.data, housing.target

# 划分测试集（必须和训练时完全一致！）
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# 2~5. 原有评估逻辑（保持不变）
# ==============================
os.makedirs('evals', exist_ok=True)

model_configs = [
    {
        'name': 'Linear Regression',
        'path': 'E:/AI_learning/models/california_housing_pipeline_v1_linear.joblib',
        'type': 'regressor'
    },
    {
        'name': 'Random Forest',
        'path': 'E:/AI_learning/models/california_housing_pipeline_v1_rf.joblib',
        'type': 'regressor'
    }
]


results = []
for config in model_configs:
    try:
        print(f"🔍 评估模型: {config['name']}")
        if not os.path.exists(config['path']):
            raise FileNotFoundError(f"模型文件不存在: {config['path']}")
        
        model = joblib.load(config['path'])
        y_pred = model.predict(X_test)
        
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        negative_count = int((y_pred < 0).sum())
        
        result = {
            'model_name': config['name'],
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'r2': round(r2, 4),
            'negative_predictions': negative_count,
            'is_business_safe': negative_count == 0
        }
        results.append(result)
        print(f"  ✅ MAE: ${mae:.2f}k, R²: {r2:.4f}, 负预测: {negative_count}")
        
    except Exception as e:
        print(f"  ❌ 评估失败: {e}")
        results.append({
            'model_name': config['name'],
            'error': str(e)
        })

# 保存报告
report_path = 'E:/AI_learning/evals/model_comparison_day14.json'
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n📊 模型对比报告已保存至: {report_path}")

# 推荐逻辑
safe_models = [r for r in results if r.get('is_business_safe', False)]
if safe_models:
    best = max(safe_models, key=lambda x: x['r2'])
    print(f"\n🏆 推荐模型: {best['model_name']} (R²={best['r2']:.4f}, 无负预测)")
else:
    print("\n⚠️ 警告：所有模型均存在负预测！")

    # 6. 生成 Markdown 报告（用于 README 或文档）
md_lines = ["# 📊 Day 14 模型 A/B 测试报告\n"]
md_lines.append("| 模型 | MAE ($k) | RMSE ($k) | R² | 负预测数 | 安全 |")
md_lines.append("|------|----------|-----------|-----|----------|------|")

for r in results:
    if 'error' not in r:
        safe_icon = "✅" if r['is_business_safe'] else "❌"
        md_lines.append(
            f"| {r['model_name']} | {r['mae']:.2f} | {r['rmse']:.2f} | {r['r2']:.4f} | {r['negative_predictions']} | {safe_icon} |"
        )


with open('E:/AI_learning/evals/model_comparison_day14.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))


print("📄 Markdown 报告已生成: E:/AI_learning/evals/model_comparison_day14.md")