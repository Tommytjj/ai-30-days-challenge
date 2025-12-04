# ====== Day 5: 决策树与可解释性AI ====== 
from sklearn.datasets import load_iris 
from sklearn.tree import DecisionTreeClassifier, plot_tree 
import matplotlib.pyplot as plt 

# 加载完整数据（4个特征） 
iris = load_iris() 
X = iris.data 
y = iris.target 
feature_names = iris.feature_names 
class_names = iris.target_names 

print("特征列表:", feature_names) 
print("花的种类:", class_names) 

# 创建决策树模型（限制深度避免过拟合） 
model = DecisionTreeClassifier( 
    max_depth=3, # 最多3层，保证可读性 
    random_state=42 ) 

# 训练模型 
model.fit(X, y) 

# 测试准确率（用全部数据，因决策树可100%拟合Iris） 
acc = model.score(X, y) 
print(f"\n模型准确率: {acc:.2%}") 

# === 关键步骤1：打印特征重要性 === 
print("\n特征重要性（越接近1越重要）:") 
for name, importance in zip(feature_names, model.feature_importances_): 
    print(f" {name}: {importance:.3f}") 
    
# === 关键步骤2：可视化决策树 === 
plt.figure(figsize=(12, 8)) 
plot_tree( 
    model, 
    feature_names=feature_names, 
    class_names=class_names, 
    filled=True, # 填充颜色表示类别 
    rounded=True, # 圆角框 
    fontsize=10 
    ) 

plt.savefig('iris_tree.png', dpi=150, bbox_inches='tight') 
print("\n✅ 决策树图已保存为: iris_tree.png") 
plt.show()




