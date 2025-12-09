# ====== Day 7: 用Flask构建 AI Web API ======
from flask import Flask,request,jsonify
import joblib
import numpy as np


# 初始化Flask应用
app = Flask(__name__)

# 在应用启动时加载模型（只加载一次）
print("正在加载AI模型")
model = joblib.load('iris_model.joblib')
iris_names = ['setosa', 'versicolor', 'virginica']
print('模型加载完成！！！')


# 定义预测接口
@app.route('/predict',methods=['POST'])
def predict():
    try:
        # 获取JSON数据，并且转化为python字典 例如:data = {"features": [5.1, 3.5, 1.4, 0.2]}
        data = request.get_json()

        # 获取数据列表，应该是长度为4的列表
        features = data['features']
    
        # 转换为模型需要的数据格式二维数组(1,4)
        X = np.array(features).reshape(1,-1)

        # 预测
        prediction = model.predict(X)[0]
        result = iris_names[prediction]

        return jsonify({
            "success": True,
            "prediction":result,
            "input_features":features
        })


    except Exception as e:
        return jsonify({
            "success": False,
            "error":str(e),
        }),400
    

# 主页（可选）
@app.route('/')
def home():
    return "<h1>Iris Flower Classifier API</h1>"
"<p>Send POST request to <code>/predict</code> with JSON: {\"features\": [sl, sw, pl, pw]}</p >"

# 启动服务器
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)