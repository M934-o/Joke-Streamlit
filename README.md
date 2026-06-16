# 笑话推荐系统 (Joke Recommender)

基于 **Item-Item 协同过滤** 的个性化笑话推荐 Web 应用。
数据集：[Jester 数据集](https://goldberg.berkeley.edu/jester-data/)。

## 功能
- 随机展示 3 个笑话，收集用户评分
- 根据用户对 3 个笑话的评分，用余弦相似度加权聚合推荐 5 个笑话
- 收集用户对推荐的反馈，归一化到 0~100% 计算满意度

## 方法
- **相似度**：Item-Item 协同过滤，Cosine 相似度
- **评分矩阵**：136 个笑话 × 7,698 用户（pivot 长格式 CSV 得到）
- **推荐逻辑**：用户对 N 个笑话评分时，对每个已评分笑话取相似度向量，按用户评分加权求和，排除已评分项后取 Top-K

## 文件结构
```
.
├── streamlit_app.py        # Streamlit Web 应用（主入口）
├── requirements.txt        # 依赖列表（已锁定版本）
├── processed_ratings.csv   # 长格式评分数据
├── Dataset4JokeSet.xlsx    # 笑话文本（158 条）
└── README.md
```

## 本地运行
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
默认访问 `http://localhost:8501`。

## 在线访问
部署在 Streamlit Community Cloud：
> 上线后填入 share.streamlit.io 给出的链接

## 实验信息
- 课程实验八
- 算法从实验七（Item-Item CF）改造而来
- 输入：用户对 3 个笑话的评分
- 输出：Top-5 推荐笑话
