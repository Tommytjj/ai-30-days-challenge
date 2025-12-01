# ====== Day 3: 训练第一个AI模型 ====== 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score 

# 1. 加载数据 n
iris = load_iris() 
X = iris.data[:, 2:]
y = iris.target

# X = iris.data # 特征：4个尺寸
# y = iris.target # 标签：0,1,2（花的种类）  


# 2. 拆分训练集（80%）和测试集（20%） 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=20 ) 


# 3. 创建模型（K近邻算法） 
model = KNeighborsClassifier(n_neighbors=3) 


# 4. 训练模型（关键！！！！！！） 
model.fit(X_train, y_train)


# 5. 在测试集上预测 
y_pred = model.predict(X_test) 


# 在预测后加： 
import matplotlib.pyplot as plt 
plt.figure(figsize=(8, 6)) 
colors = ['red' if pred != true else 'blue' for pred, true in zip(y_pred, y_test)] 
plt.scatter(X_test[:, 0], X_test[:, 1], c=colors, alpha=0.8) 
plt.title('预测结果：蓝色=正确，红色=错误',fontproperties='SimHei') 
plt.savefig('iris_prediction_errors.png') 
plt.show()

# 6. 计算准确率 
acc = accuracy_score(y_test, y_pred) 
print(f"模型准确率: {acc:.2%} ({sum(y_pred == y_test)}/{len(y_test)} 题答对)")