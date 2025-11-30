# ====== Day 2: 鸢尾花数据探索 ====== 
# 导入工具 
import pandas as pd 
from sklearn.datasets import load_iris 
import matplotlib.pyplot as plt 

# 1. 加载数据 print("正在加载鸢尾花数据...") 
iris = load_iris()# 这是一个内置的小数据集  


# 2. 转成表格（DataFrame） 
df = pd.DataFrame( data=iris.data, columns=iris.feature_names # 列名：花萼长/宽、花瓣长/宽 
    )
 
df['species'] = iris.target # 添加数字标签（0,1,2） 

# 3. 把数字换成花的名字（更易读） 
df['species'] = df['species'].map({ 0: 'setosa', 1: 'versicolor', 2: 'virginica' }) 

# 4. 打印基本信息 
print("\n✅ 数据总共有", df.shape[0], "行，", df.shape[1], "列") 
print("\n前5行数据：") 
print(df.head()) 
print("\n 每种花的数量：") 
print(df['species'].value_counts())


# ====== 第 2 步：画图 ====== 
import matplotlib.pyplot as plt 

print("\n正在绘制花瓣散点图...") 


print(df.groupby('species'))

# 创建一个新图表 
plt.figure(figsize=(8, 6)) # 分别画出三种花（按 species 分组） 

# 核心代码，执行绘图操作
for species_name, group in df.groupby('species'):# df.groupby('species')：将数据按照类别进行分组  
    plt.scatter(x=group['petal length (cm)'], 
                y=group['petal width (cm)'], 
                label=species_name, 
                alpha=0.8, # 透明度 
                s=60 # 点的大小 
                ) 
# 添加坐标轴标签和标题 
plt.xlabel('花瓣长度 (cm)',fontproperties='SimHei')
plt.ylabel('花瓣宽度 (cm)',fontproperties='SimHei') 
plt.title('鸢尾花种类 vs 花瓣尺寸',fontproperties='SimHei') 
plt.legend() # 显示图例（红=setosa, 绿=versicolor...） 
plt.grid(True) # 显示网格线 
# 保存图片到当前目录（比如桌面） 
plt.savefig('iris_scatter.png', dpi=150, bbox_inches='tight') 
print("✅ 图片已保存为：iris_scatter.png") 
# 尝试弹出窗口显示（如果没弹出，直接去文件夹找 .png） 
plt.show()


