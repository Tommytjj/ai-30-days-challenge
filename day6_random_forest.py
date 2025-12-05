# ====== Day 6: 随机森林 —— 集成学习的力量 ====== 
from sklearn.datasets import load_iris 
from sklearn.ensemble import RandomForestClassifier 
import matplotlib.pyplot as plt 
import numpy as np 

# 加载数据 
iris = load_iris()
X = iris.data 
y = iris.target 
feature_names = iris.feature_names 
print("特征列表:", feature_names) 

# 创建随机森林模型（100棵树） 
model = RandomForestClassifier( 
    n_estimators=100, # 树的数量 
    random_state=42, 
    oob_score=True # 启用袋外评估（无需单独测试集） 
    ) 


# 训练模型 
model.fit(X, y) 

# 准确率（使用袋外分数，更可靠） 
print(f"\n袋外准确率 (OOB Score): {model.oob_score_:.2%}") 


# === 关键步骤：特征重要性 === 
importances = model.feature_importances_ 
print("\n随机森林特征重要性:") 
for name, imp in zip(feature_names, importances): 
    print(f" {name}: {imp:.3f}") 

# === 可视化：特征重要性柱状图（可选但推荐）=== 
plt.figure(figsize=(8, 5)) 
y_pos = np.arange(len(feature_names)) 
plt.barh(y_pos, importances, color='skyblue') 
plt.yticks(y_pos, feature_names) 
plt.xlabel('重要性',fontproperties='SimHei') 
plt.title('随机森林特征重要性',fontproperties='SimHei') 
plt.tight_layout() 
plt.savefig('feature_importance.png', dpi=150) 
print("\n✅ 特征重要性图已保存为: feature_importance.png") 
plt.show()

