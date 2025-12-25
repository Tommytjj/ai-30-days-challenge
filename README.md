My AI Journey 
Day 1:环境搭建
Day 2:鸢尾花数据分析与可视化
Day 3:训练第一个机器学习模型
Day 4:模型评估与交叉验证
Day 5:决策树与可解释AI
Day 6:随机森林：集成学习的力量
Day 7:模型持久化
Day 8:Flask AI Web API
Day 9:前端网页 + AI后端联调
Day 10:云函数部署(没完成)
Day 11:模型评估与优化
Day 12:特征工程与 Pipeline
Day 13:回归任务实战  
  - 初始使用 LinearRegression，出现负数预测  
  - 改用 RandomForestRegressor，结果合理
  - 虽然 LinearRegression 的 R² (0.618) 略高于 RandomForest (0.580)，
  - 但线性模型会输出负房价（违反业务逻辑），
  - 因此选择 RandomForest 作为 v1 回归模型 —— **可靠性优先于微小指标优势**。
Day 14: 模型版本管理与 A/B 测试（支持离线数据）
Day 15: 超参数调优（RandomizedSearchCV），RF v2 R² 提升至 0.4109
