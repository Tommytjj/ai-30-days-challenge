# ==============================
# 🌟 Day 16：统一预测 API（带逐行中文注释）
# 功能：通过一个接口 /predict 支持 鸢尾花分类 和 房价预测
# ==============================

# 第一步：导入必要的工具包
from fastapi import FastAPI, HTTPException  # FastAPI 用于创建 Web 接口，HTTPException 用于返回错误
from pydantic import BaseModel, Field      # Pydantic 用于校验用户输入的数据格式
from typing import List, Literal           # 用于定义“只能是某些值”的类型（比如 task_type 只能是 "iris" 或 "housing"）
import joblib                              # 用于加载你之前保存的 .joblib 模型
import os                                  # 用于处理文件路径（跨平台兼容）

# 第二步：创建 FastAPI 应用对象
# 这个 app 就是你的“服务器”，所有接口都注册在它上面
app = FastAPI(
    title="AI 30 Days Challenge - Prediction API",
    description="一个统一的 AI 预测服务，支持鸢尾花分类和加州房价预测",
    version="1.0"
)

# 第三步：定义模型存放的目录
# __file__ 是当前文件（main.py）的路径
# os.path.dirname(__file__) → 得到 api/ 目录
# 再往上一层（..）就是 E:\AI_learning\
# 所以 MODEL_DIR = "E:\AI_learning\models"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# 第四步：启动时自动加载两个模型（只加载一次，提高速度）
# 注意：如果模型文件不存在，程序会报错！所以你要确认文件名正确
try:
    print("🔍 正在加载鸢尾花分类模型...")
    # 加载你在 Day 11 保存的逻辑回归模型
    iris_model = joblib.load(os.path.join(MODEL_DIR, "iris_pipeline_v2.joblib"))
    
    print("🔍 正在加载房价回归模型...")
    # 加载你在 Day 15 调优后的随机森林模型（推荐用 v2）
    housing_model = joblib.load(os.path.join(MODEL_DIR, "regressor_v2_rf_tuned.joblib"))
    
    print("✅ 所有模型加载成功！")
except FileNotFoundError as e:
    # 如果文件没找到，打印具体缺失的文件
    print(f"❌ 模型文件未找到: {e}")
    print("请检查 models/ 目录下是否有以下文件：")
    print("  - iris_pipeline_v2.joblib")
    print("  - regressor_v2_rf_tuned.joblib")
    # 设置为 None，后续请求会返回 500 错误
    iris_model = None
    housing_model = None
except Exception as e:
    print(f"❌ 模型加载出错: {e}")
    iris_model = None
    housing_model = None

# 第五步：定义用户请求的数据格式（用 Pydantic）
# 当用户发 POST 请求时，必须符合这个结构
class PredictionRequest(BaseModel):
    # task_type 只能是 "iris" 或 "housing"，不能是别的
    task_type: Literal["iris", "housing"] = Field(
        ...,  # ... 表示这个字段是必填的
        description="任务类型：'iris' 表示鸢尾花分类，'housing' 表示房价预测"
    )
    # features 是一个浮点数列表，比如 [5.1, 3.5, 1.4, 0.2]
    features: List[float] = Field(
        ...,
        description="特征列表。鸢尾花需要 4 个，房价需要 8 个。"
    )

# 第六步：定义返回给用户的数据格式
class PredictionResponse(BaseModel):
    task_type: str                     # 返回任务类型
    prediction: float | str            # 分类返回字符串（如 "setosa"），回归返回数字（如 2.96）
    label: str | None = None          # 额外信息：分类时返回人类可读标签，回归时为 null

# 第七步：定义核心预测接口
# 当用户访问 POST /predict 时，执行这个函数
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):  # request 自动被 Pydantic 校验
    """
    统一预测接口：
    - 如果 task_type 是 "iris"，调用鸢尾花模型
    - 如果 task_type 是 "housing"，调用房价模型
    """
    
    # =============== 处理鸢尾花分类 ===============
    if request.task_type == "iris":
        # 检查模型是否加载成功
        if iris_model is None:
            raise HTTPException(status_code=500, detail="鸢尾花模型未加载，请检查文件")
        
        # 检查特征数量是否为 4（鸢尾花有 4 个特征）
        if len(request.features) != 4:
            raise HTTPException(
                status_code=400,
                detail=f"鸢尾花需要 4 个特征，但收到了 {len(request.features)} 个"
            )
        
        # 调用模型预测
        # 注意：predict() 需要二维数组，所以用 [request.features] 包一层
        pred_class_index = iris_model.predict([request.features])[0]  # 得到数字：0, 1, 或 2
        
        # 把数字转成花的名字（顺序必须和 Day 11 一致！）
        species_map = ["setosa", "versicolor", "virginica"]
        predicted_species = species_map[int(pred_class_index)]
        
        # 返回结果
        return PredictionResponse(
            task_type="iris",
            prediction=predicted_species,  # 直接返回名字更友好
            label=predicted_species
        )

    # =============== 处理房价预测 ===============
    elif request.task_type == "housing":
        if housing_model is None:
            raise HTTPException(status_code=500, detail="房价模型未加载，请检查文件")
        
        if len(request.features) != 8:
            raise HTTPException(
                status_code=400,
                detail=f"加州房价需要 8 个特征，但收到了 {len(request.features)} 个"
            )
        
        # 调用模型预测
        predicted_price = housing_model.predict([request.features])[0]  # 单位：千美元
        
        # ⚠️ 关键安全措施：确保房价不为负（你 Day 13 学到的教训！）
        predicted_price = max(0.0, predicted_price)
        
        # 保留两位小数
        predicted_price = round(float(predicted_price), 2)
        
        return PredictionResponse(
            task_type="housing",
            prediction=predicted_price,
            label=None
        )

    # =============== 其他情况（理论上不会发生，因为 Literal 限制了）===============
    else:
        raise HTTPException(status_code=400, detail="task_type 必须是 'iris' 或 'housing'")