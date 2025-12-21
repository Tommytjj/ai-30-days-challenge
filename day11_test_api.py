# test_api.py —— 本地模拟云函数调用
import json
import joblib

# 加载你刚训练的模型
model = joblib.load('iris_model_v1.joblib')

def simulate_cloud_function(features):
    """模拟云函数的 main_handler"""
    try:
        # 预测
        pred = model.predict([features])[0]
        species = ['setosa', 'versicolor', 'virginica'][pred]
        return {'result': f'这是一朵 {species}！'}
    except Exception as e:
        return {'error': str(e)}

# 测试几个例子
test_cases = [
    [5.1, 3.5, 1.4, 0.2],   # setosa
    [6.2, 2.8, 4.8, 1.8],   # virginica
    [5.5, 2.4, 3.7, 1.0]    # versicolor
]

for i, features in enumerate(test_cases, 1):
    response = simulate_cloud_function(features)
    print(f"测试 {i}: 输入 {features} → {response}")
    