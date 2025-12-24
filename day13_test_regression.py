# test_regression.py
import joblib

# 加载随机森林模型
pipe = joblib.load('california_housing_pipeline_v1_rf.joblib')

# 加载回归模型
# pipe = joblib.load('california_housing_pipeline_v1_linear.joblib')

# 模拟一条房屋数据: [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
sample_house = [8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]

predicted_price = pipe.predict([sample_house])[0]
print(f"🔮 预测房价中位数: ${predicted_price:.2f} 万美元")