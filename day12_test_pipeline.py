# test_pipeline.py
import joblib

pipe = joblib.load('iris_pipeline_v2.joblib')
pred = pipe.predict([[5.1, 3.5, 1.4, 0.2]])
species = ['setosa', 'versicolor', 'virginica'][pred[0]]
print("预测结果:", species)  # 应输出 setosa