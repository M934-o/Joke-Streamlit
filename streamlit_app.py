"""Joke Recommendation Web App
Lab 8 - 基于协同过滤的笑话推荐系统

实现思路（取自实验七 Item-Item CF，已适配 Streamlit 部署）：
  1) 读取 processed_ratings.csv（长格式）与 Dataset4JokeSet.xlsx（笑话文本）
  2) 将长格式 pivot 为 笑话×用户 评分矩阵，缺失值填 0
  3) 对该矩阵做 cosine_similarity，得到 笑话×笑话 相似度矩阵
  4) 推荐函数：用户对 N 个笑话评分时，对每个已评分笑话取相似度向量，
     按用户评分加权求和得到综合得分，排除已评分项后取 Top-K
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity


RATINGS_FILE = "processed_ratings.csv"
JOKES_FILE = "Dataset4JokeSet.xlsx"
RATING_MIN, RATING_MAX = -42.28, 28.58  # 与数据集中实际评分范围一致


# ---------- 数据加载 ----------

@st.cache_data
def load_ratings(path: str = RATINGS_FILE) -> pd.DataFrame:
    """读取长格式评分数据。"""
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_jokes(path: str = JOKES_FILE) -> pd.DataFrame:
    """读取笑话文本（无表头），按 joke_id = 行号(1-indexed) 建立映射。"""
    df = pd.read_excel(path, header=None)
    df.columns = ["joke_text"]
    df.index = df.index + 1  # joke_id 从 1 开始
    df.index.name = "joke_id"
    return df


def get_joke_text(joke_id: int, jokes: pd.DataFrame) -> str:
    if joke_id in jokes.index:
        return str(jokes.loc[joke_id, "joke_text"])
    return "(无文本)"


# ---------- 相似度矩阵 ----------

@st.cache_resource
def build_similarity(ratings: pd.DataFrame):
    """构建 Item-Item 相似度矩阵与 笑话×用户 评分矩阵。

    Returns
    -------
    sim_df   : DataFrame (n_jokes × n_jokes)，index/columns 为 joke_id
    matrix   : DataFrame (n_jokes × n_users)，index 为 joke_id
    """
    matrix = ratings.pivot_table(
        index="joke_id", columns="user_id", values="rating", fill_value=0
    )
    sim = cosine_similarity(matrix.values)
    sim_df = pd.DataFrame(sim, index=matrix.index, columns=matrix.index)
    return sim_df, matrix


# ---------- 推荐函数（实验七 → 实验八改造） ----------

def recommend_topk(
    user_ratings: dict[int, float],
    sim_df: pd.DataFrame,
    top_k: int = 5,
) -> pd.Series:
    """根据用户对若干笑话的评分，综合推荐 Top-K 笑话。

    Parameters
    ----------
    user_ratings : {joke_id: rating}，键是 joke_id，值是用户评分
    sim_df       : 笑话相似度矩阵
    top_k        : 推荐数量

    Returns
    -------
    pd.Series，index 为推荐 joke_id，value 为综合得分，按得分降序
    """
    score = pd.Series(0.0, index=sim_df.index)
    for jid, r in user_ratings.items():
        if jid in sim_df.index and r != 0:
            score = score.add(sim_df.loc[jid] * r, fill_value=0.0)

    # 排除已评分笑话
    score = score.drop(labels=list(user_ratings.keys()), errors="ignore")
    # 排除相似度全 0 的（无法推荐）
    score = score[score > 0]
    return score.sort_values(ascending=False).head(top_k)


def normalize_to_percent(rating: float) -> float:
    """把评分线性归一化到 0~100% 区间。"""
    if RATING_MAX == RATING_MIN:
        return 0.0
    return (rating - RATING_MIN) / (RATING_MAX - RATING_MIN) * 100.0


# ---------- Streamlit 界面 ----------

st.set_page_config(page_title="Joke Recommender", page_icon="😄", layout="wide")


def init_session():
    defaults = {
        "phase": "init",
        "seed_jokes": [],          # 初始 3 个让用户评分的笑话
        "initial_ratings": {},     # {joke_id: rating}
        "recommendations": None,   # pd.Series index=joke_id, value=score
        "rec_ratings": {},         # {joke_id: rating}
        "satisfaction": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def pick_random_jokes(ratings: pd.DataFrame, jokes: pd.DataFrame, n: int = 3):
    """从同时存在文本与评分的 joke_id 中随机选 n 个。"""
    valid_ids = sorted(set(ratings["joke_id"]) & set(jokes.index))
    return list(np.random.choice(valid_ids, size=n, replace=False))


# 侧边栏：项目信息
with st.sidebar:
    st.title("😄 Joke Recommender")
    st.markdown("**实验八 · 个性化笑话推荐**")
    st.markdown("---")
    st.markdown("**方法**：Item-Item 协同过滤")
    st.markdown("**相似度**：Cosine")
    st.markdown("**数据集**：Jester 1.4 万条评分")
    st.markdown("---")
    st.markdown("**流程**：")
    st.markdown("1. 给 3 个随机笑话打分")
    st.markdown("2. 系统推荐 5 个笑话")
    st.markdown("3. 给推荐打分，计算满意度")
    if st.button("🔄 重新开始", use_container_width=True, key="restart_btn"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

init_session()
ratings = load_ratings()
jokes = load_jokes()
sim_df, matrix = build_similarity(ratings)

# 关键：先用“稳定 key”预创建 3 个 seed slider 和 5 个 rec slider 的会话状态。
# 否则 phase 切换、组件被卸载时，Streamlit 会因 key 缺失而抛 KeyError。
# 这里只初始化默认值。后续用户拖动 slider 改变的值会写回同一 key。
for i in range(3):
    st.session_state.setdefault(f"seed_rating_{i}", 0.0)
for i in range(5):
    st.session_state.setdefault(f"rec_rating_{i}", 0.0)

st.title("个性化笑话推荐系统")
st.caption(
    f"数据：{len(ratings):,} 条评分 · "
    f"{ratings['user_id'].nunique():,} 用户 · "
    f"{ratings['joke_id'].nunique():,} 笑话 · "
    f"评分范围 [{RATING_MIN:.2f}, {RATING_MAX:.2f}]"
)

# ===== 阶段 1：开始 =====
if st.session_state.phase == "init":
    st.subheader("Step 1 · 给 3 个随机笑话打分")
    st.write("点击下方按钮开始，系统会随机展示 3 个笑话。")
    if st.button("开始评分", type="primary", key="start_btn"):
        st.session_state.seed_jokes = pick_random_jokes(ratings, jokes, 3)
        st.session_state.phase = "rate3"
        st.rerun()

# ===== 阶段 2：给 3 个笑话评分 =====
elif st.session_state.phase == "rate3":
    st.subheader("Step 1 · 给 3 个随机笑话打分")
    st.progress(0.4, text="请为下面 3 个笑话打分（-42.28 = 非常不喜欢, 28.58 = 非常喜欢）")
    cols = st.columns(3)
    new_ratings = {}
    for i, jid in enumerate(st.session_state.seed_jokes):
        with cols[i]:
            st.markdown(f"**Joke #{int(jid)}**")
            st.markdown(f"> {get_joke_text(int(jid), jokes)}")
            st.slider(
                "你的评分",
                min_value=float(RATING_MIN),
                max_value=float(RATING_MAX),
                step=0.1,
                key=f"seed_rating_{i}",
            )
        # 从 session_state 读取评分（slider 的 value 与 key 绑定）
        new_ratings[int(jid)] = float(st.session_state[f"seed_rating_{i}"])
    st.session_state.initial_ratings = new_ratings

    c1, c2 = st.columns([1, 5])
    with c1:
        if st.button("🎯 获取推荐", type="primary", use_container_width=True, key="get_recs_btn"):
            recs = recommend_topk(new_ratings, sim_df, top_k=5)
            if len(recs) == 0:
                st.error("没有可推荐的笑话，请尝试调整评分。")
            else:
                st.session_state.recommendations = recs
                st.session_state.phase = "rec"
                st.rerun()
    with c2:
        st.info(f"已评分：{list(new_ratings.keys())}")

# ===== 阶段 3：展示推荐并打分 =====
elif st.session_state.phase == "rec":
    recs: pd.Series = st.session_state.recommendations
    st.subheader("Step 2 · 为推荐的 5 个笑话打分")
    st.progress(0.7, text="下面是基于你的初始评分推荐出的笑话，请为每个打分")

    new_rec_ratings = {}
    for i, (jid, score) in enumerate(recs.items(), start=1):
        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"**推荐 #{i} · Joke #{int(jid)}** （综合得分 {score:.3f}）")
                st.markdown(f"> {get_joke_text(int(jid), jokes)}")
            with c2:
                st.slider(
                    "你的评分",
                    min_value=float(RATING_MIN),
                    max_value=float(RATING_MAX),
                    step=0.1,
                    key=f"rec_rating_{i-1}",
                    label_visibility="collapsed",
                )
            new_rec_ratings[int(jid)] = float(st.session_state[f"rec_rating_{i-1}"])
    st.session_state.rec_ratings = new_rec_ratings

    if st.button("📊 计算满意度", type="primary", key="calc_sat_btn"):
        if not new_rec_ratings:
            st.warning("请先给推荐笑话打分")
        else:
            percents = [normalize_to_percent(v) for v in new_rec_ratings.values()]
            st.session_state.satisfaction = float(np.mean(percents))
            st.session_state.phase = "done"
            st.rerun()

# ===== 阶段 4：满意度 =====
elif st.session_state.phase == "done":
    st.subheader("Step 3 · 满意度报告")
    sat = st.session_state.satisfaction
    c1, c2, c3 = st.columns(3)
    c1.metric("推荐满意度", f"{sat:.1f}%")
    c2.metric("已评分推荐数", f"{len(st.session_state.rec_ratings)}")
    c3.metric("综合得分均值", f"{st.session_state.recommendations.mean():.3f}")
    st.progress(min(sat / 100.0, 1.0))

    with st.expander("查看本次推荐详情", expanded=True):
        for jid, rating in st.session_state.rec_ratings.items():
            pct = normalize_to_percent(rating)
            st.markdown(
                f"- **Joke #{int(jid)}** · 你的评分 `{rating:+.2f}` → "
                f"归一化满意度 **{pct:.1f}%**"
            )

    if st.button("🔁 再来一轮", type="primary"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
