import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st

st.set_page_config(page_title="球鞋買價分析", layout="wide")

# ==================== 主標題 ====================
st.title("👟 球鞋買價分析")

# ==================== 側邊欄設定 ====================
st.sidebar.header("⚙️ 數據與模型設定")
uploaded_file = st.sidebar.file_uploader("上傳成交歷史 CSV", type=["csv"])
current_ask = st.sidebar.number_input(
    "目前即時賣價 (即買價)", value=402.00, step=1.0
)

# 讀取 CSV 或使用預設數據
if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(
            uploaded_file, header=None, dtype=str, on_bad_lines="skip"
        )
        price_col = (
            3
            if df_raw.shape[1] > 3
            else (df_raw.shape[1] - 1 if df_raw.shape[1] > 1 else 0)
        )
        prices_series = pd.to_numeric(
            df_raw.iloc[:, price_col]
            .astype(str)
            .str.replace(r"[^\d.]", "", regex=True),
            errors="coerce",
        ).dropna()
        prices = prices_series.values
        st.sidebar.success(f"✅ 成功讀取 {len(prices)} 筆歷史成交價！")
    except Exception as e:
        st.sidebar.error(f"❌ 讀取失敗：{e}")
        prices = np.array([])
else:
    st.sidebar.info("💡 未上傳 CSV，使用預設數據")
    prices = np.array(
        [
            542, 472, 552, 446, 451, 467, 523, 447, 446, 384,
            474, 478, 435, 685, 575, 474, 472, 495, 478, 460,
            474, 458, 495, 502, 487, 545, 547, 549, 448, 450,
            442, 587, 653, 422, 514, 337, 489, 449, 422, 450,
            473, 489, 449, 507, 509, 473, 449, 554, 384, 532,
        ],
        dtype=float,
    )

if len(prices) > 0:
    # ==================== 統計量計算 ====================
    mean_val = float(np.mean(prices))
    std_val = float(np.std(prices, ddof=1))
    p25 = float(np.percentile(prices, 25))
    p50 = float(np.percentile(prices, 50))
    p75 = float(np.percentile(prices, 75))

    p15 = float(np.percentile(prices, 15))
    p85 = float(np.percentile(prices, 85))

    ideal_bid = p25
    over_bid = p75
    min_price = int(np.min(prices))
    max_price = int(np.max(prices))

    # 建立左右兩欄佈局
    col_left, col_right = st.columns([1.1, 1], gap="large")

    # ==================== 左側欄位 ====================
    with col_left:
        # 1. 統計指標區塊
        st.subheader("📊 近期成交統計指標")

        m1, m2 = st.columns(2)
        m1.metric("平均價格 (μ)", f"{mean_val:.1f} 美元")
        m2.metric("價格波動度/標準差 (s)", f"{std_val:.1f} 美元")

        m3, m4 = st.columns(2)
        m3.metric("中位數 (P50)", f"{p50:.1f} 美元")

        with m4:
            st.markdown(
                f"""
                <div style="font-size: 14px; color: #31333F; margin-bottom: 4px;">近期最低/最高</div>
                <div style="font-size: 24px; font-weight: bold; line-height: 1.2; color: #111;">
                    {min_price}美元 - {max_price}美元
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        # 2. 直方圖區塊
        st.subheader("📈 近期成交價格直方圖")

        fig = go.Figure()

        # 固定組寬為 25 美元，起點從 325 或 350 開始對齊整數區間
        bin_size = 25
        start_val = 325

        # 70% 主要買價區間背景 (黃色半透明)
        fig.add_vrect(
            x0=p15,
            x1=p85,
            fillcolor="rgba(240, 230, 140, 0.35)",
            layer="below",
            line_width=0,
        )

        # 直方圖設定 (組寬固定 25)
        fig.add_trace(
            go.Histogram(
                x=prices,
                xbins=dict(start=start_val, size=bin_size),
                autobinx=False,
                name="成交次數 (頻率)",
                marker_color="rgba(100, 149, 237, 0.75)",
                marker_line_color="#2b5c8f",
                marker_line_width=1.2,
            )
        )

        # KDE 趨勢擬合線
        kde = stats.gaussian_kde(prices, bw_method=0.33)
        x_grid = np.linspace(325, 625, 300)
        y_kde = kde(x_grid) * len(prices) * bin_size

        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=y_kde,
                mode="lines",
                name="趨勢擬合線",
                line=dict(color="#e74c3c", width=3),
            )
        )

        # 垂直標註線 (現買價、理想買價、平均價、溢買價)
        fig.add_vline(
            x=current_ask,
            line_dash="solid",
            line_color="#2980b9",
            line_width=2.5,
            annotation_text=f"現買價<br>${current_ask:.0f}",
            annotation_position="top left",
            annotation_font_color="#2980b9",
        )

        fig.add_vline(
            x=ideal_bid,
            line_dash="dash",
            line_color="#27ae60",
            line_width=2,
            annotation_text=f"理想買價<br>${ideal_bid:.0f}",
            annotation_position="top left",
            annotation_font_color="#27ae60",
        )

        fig.add_vline(
            x=mean_val,
            line_dash="dot",
            line_color="#8e44ad",
            line_width=2,
            annotation_text=f"平均價<br>${mean_val:.0f}",
            annotation_position="top",
            annotation_font_color="#8e44ad",
        )

        fig.add_vline(
            x=over_bid,
            line_dash="dash",
            line_color="#c0392b",
            line_width=2,
            annotation_text=f"溢買價<br>${over_bid:.0f}",
            annotation_position="top right",
            annotation_font_color="#c0392b",
        )

        center_x = (p15 + p85) / 2
        fig.add_annotation(
            x=center_x,
            y=0.05,
            xref="x",
            yref="paper",
            text="70% 主要買價區間",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=11, color="#333333"),
        )

        # X軸以50為刻度，Y軸以2為刻度 (與目標圖一致)
        fig.update_layout(
            title_text="",
            xaxis=dict(
                title="成交金額 (USD) ➔ 越往右代表買得越貴",
                dtick=50,
                tick0=350,
            ),
            yaxis=dict(
                title="頻率 (成交筆數)",
                tick0=0,
                dtick=2,
            ),
            height=430,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=True,
            legend=dict(x=0.75, y=0.95),
        )

        st.plotly_chart(fig, use_container_width=True)

    # ==================== 右側欄位 ====================
    with col_right:
        st.subheader("🎯 目前四大價格明細表")

        tag_df = pd.DataFrame(
            {
                "價格類型": [
                    "現買價 (Ask)",
                    "理想買價 (P25)",
                    "溢買價格 (P75)",
                    "70% 主要買價區間",
                ],
                "金額 (USD)": [
                    f"US$ {current_ask:.1f}",
                    f"US$ {ideal_bid:.1f}",
                    f"US$ {over_bid:.1f}",
                    f"US$ {p15:.1f} - {p85:.1f}",
                ],
                "分布位置與解讀": [
                    "當前賣家掛單價，低於理想買價即可秒殺",
                    "前 25% 低價區，最具 CP 值",
                    "超過 75% 歷史價，屬高價追高區",
                    "涵蓋歷史 70% 的集中成交區間",
                ],
            }
        )
        st.dataframe(tag_df, use_container_width=True, hide_index=True)

        st.write("")
        st.write("")

        # 買家出價策略建議
        if current_ask <= ideal_bid:
            strategy_status = "🔥 極限推薦秒殺"
            strategy_color = "#27ae60"
            strategy_desc = f"目前賣家掛單價（${current_ask:.0f}）已低於理想買價 P25（${ideal_bid:.0f}），位於歷史極低價區，建議直接點擊購買秒殺！"
        elif current_ask <= mean_val:
            strategy_status = "✅ 合理價格區間"
            strategy_color = "#2980b9"
            strategy_desc = f"目前賣家掛單價（${current_ask:.0f}）低於市場平均價（${mean_val:.0f}）。若急著入手可直接購買，或掛單在 **${ideal_bid:.0f} (P25)** 嘗試撿便宜。"
        else:
            strategy_status = "⚠️ 偏高/建議掛單等待"
            strategy_color = "#c0392b"
            strategy_desc = f"目前賣家掛單價（${current_ask:.0f}）高於平均價。建議不要直接購買，改以 **${ideal_bid:.0f} ~ ${p50:.0f}** 進行出價掛單（Bid）等待賣家撮合。"

        st.markdown(
            f"""
            <div style="background-color: #f8f9fa; border-left: 5px solid {strategy_color}; padding: 14px 18px; border-radius: 4px; margin-top: 10px;">
                <div style="font-size: 15px; font-weight: bold; color: #111; margin-bottom: 6px;">
                    💡 買家出價策略建議：<span style="color: {strategy_color};">{strategy_status}</span>
                </div>
                <div style="font-size: 13.5px; color: #444; line-height: 1.6;">
                    {strategy_desc}<br>
                    <small style="color: #777;">（資料參考：建議出價以百分位數 P25 (${ideal_bid:.0f}) 為核心基準）</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
