# ====== Day 7: 模型持久化 —— 保存与加载AI模型 ====== 
from sklearn.datasets import load_iris 
from sklearn.ensemble import RandomForestClassifier 
import joblib # 推荐用于 sklearn 模型（比 pickle 更高效） 

# === 第一步：训练模型 === 
print("🔧 正在训练随机森林模型...") 
iris = load_iris() 
X, y = iris.data, iris.target 
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X, y) 


# 保存前测试一个预测 
sample = X[0].reshape(1, -1) # 第一朵花 
pred_before = model.predict(sample) 
print(f"保存前预测: {iris.target_names[pred_before[0]]}") 


# === 第二步：保存模型到文件 === 
joblib.dump(model, 'iris_model.joblib') 
print("✅ 模型已保存为: iris_model.joblib") 


# === 第三步：加载模型 === 
print("\n📥 正在从文件加载模型...") 
loaded_model = joblib.load('iris_model.joblib') 



# 用加载的模型做同样预测 
pred_after = loaded_model.predict(sample) 
print(f"📤 加载后预测: {iris.target_names[pred_after[0]]}") 


# 验证两者一致 
if pred_before == pred_after: 
    print("\n✅ 模型保存与加载成功！预测结果一致。") 
else: 
    print("\n出错了！预测结果不一致。")


# 在加载模型后，加这段： 
print("\n请输入一朵花的4个尺寸（用空格分隔）：") 
# 示例输入：5.1 3.5 1.4 0.2
user_input = input("例如 '5.1 3.5 1.4 0.2' → ") 
features = list(map(float, user_input.split())) 
prediction = loaded_model.predict([features]) 
print(f"🤖 AI 预测：这是一朵 {iris.target_names[prediction[0]]}！")


