# ====== Day 4: 模型评估与交叉验证 ====== 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.neighbors import KNeighborsClassifier 
import numpy as np 

# 加载数据（只用花瓣特征） 
iris = load_iris() 
X = iris.data[:, 2:] # 仅花瓣长度和宽度 
y = iris.target 


print("数据形状:", X.shape) 
print("目标分布:", np.bincount(y)) # 每类50朵 




# === 实验1：不同随机种子下的准确率 === 
print("实验1：不同 random_state 下的准确率") 
model = KNeighborsClassifier(n_neighbors=3) 
for seed in [0, 1, 2, 3, 42]: 
    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.2, random_state=seed ) 
    model.fit(X_train, y_train) 
    acc = model.score(X_test, y_test) 
    print(f" Seed {seed:2d}: 准确率 = {acc:.2%}") 




# === 实验2：5折交叉验证（更可靠！）=== 
print("实验2：5折交叉验证") 
cv_scores = cross_val_score(model, X, y, cv=5) 
print(f" 平均准确率: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}") 
print(f" 各折得分: {[f'{s:.2%}' for s in cv_scores]}") 




# === 实验3：尝试 K=1（可能过拟合）=== 
print("实验3：使用 K=1（高风险过拟合）") 
overfit_model = KNeighborsClassifier(n_neighbors=1) 
overfit_scores = cross_val_score(overfit_model, X, y, cv=5) 
print(f" K=1 平均准确率: {overfit_scores.mean():.2%} ± {overfit_scores.std():.2%}")


