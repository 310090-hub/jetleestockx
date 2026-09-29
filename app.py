import os
import urllib.request
import platform
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from scipy.stats import gaussian_kde

# --- 自動下載並設定雲端中文字型 (思源黑體) ---
font_filename = "NotoSansTC-Regular.otf"
if not os.path.exists(font_filename):
    try:
        font_url = "https://github.com/google/fonts/raw/main/ofl/notosanstc/NotoSansTC-Regular.otf"
        urllib.request.urlretrieve(font_url, font_filename)
    except Exception:
        pass

if os.path.exists(font_filename):
    fm.fontManager.addfont(font_filename)
    mpl.rc('font', family='Noto Sans TC')
else:
    system = platform.system()
    if system == 'Windows':
        mpl.rc('font', family='Microsoft JhengHei')
    elif system == 'Darwin':
        mpl.rc('font', family='PingFang HK')

mpl.rcParams['axes.unicode_minus'] = False

# --- 網頁配置 ---
st.set_page_config(page_title="球鞋買價分析", layout="wide")
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 95%; }
        div[data-testid="stMetricValue"] { font-size: 26px !important; }
        div[data-testid="stMetricLabel"] { font-size: 14px !important; color: #555 !important; }
        hr { margin: 1.2rem 0 !important; }
    </style>
""", unsafe_allow_html=True)

# ================= 側邊欄 =================
with st.sidebar:
    st.markdown("### ⚙️ 數據與模型設定")
    uploaded_file = st.file_uploader("上傳成交歷史 CSV", type=['csv'])
    ask_price = st.number_input("目前開啟購買價 (詢問)", value=406.0, step=1.0)
    
    if uploaded_file is None:
        st.info("💡 未上傳CSV，使用預設50筆數據")

# ================= 處理數據 =================
try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)
        price_col_index = df.columns[-1] 
        raw_prices = df[price_col_index].astype(str)
        cleaned_prices = (raw_prices
                          .str.replace('US$', '', regex=False)
                          .str.replace('$', '', regex=False)
                          .str.replace(',', '', regex=False)
                          .str.strip())
        prices = pd.to_numeric(cleaned_prices, errors='coerce').dropna().values
        prices = prices[:50] 
    else:
        np.random.seed(42)
        prices = np.random.normal(loc=259.2, scale=24.3, size=50)
        prices = np.clip(prices, 220, 370)
        
    if len(prices) == 0:
        st.error("資料無效！")
        st.stop()
        
    mean_price = np.mean(prices)
    median_price = np.median(prices)
    std_price = np.std(prices)
    min_price = np.min(prices)
    max_price = np.max(prices)
    p15, p25, p50, p75, p85 = np.percentile(prices, [15, 25, 50, 75, 85])

    # ================= 主畫面 =================
    st.markdown("## 👟 球鞋買價分析")
    
    col_left, col_right = st.columns([5, 5], gap="large")
    
    # ----------------- 左側 -----------------
    with col_left:
        st.markdown("### 📊 近期成交統計指標")
        
        c1, c2 = st.columns(2)
        c1.metric("平均價格 (μ)", f"{mean_price:.1f} 美元")
        c2.metric("價格波動度/標準差 (s)", f"{std_price:.1f} 美元")
        
        c3, c4 = st.columns(2)
        c3.metric("中位數 (P50)", f"{median_price:.1f} 美元")
        c4.markdown(f"""
            <div style='font-size: 14px; color: #555;'>近期最低/最高</div>
            <div style='font-size: 24px; font-weight: bold; margin-top: 5px; color: #000000;'>
                {min_price:.0f}美元 - {max_price:.0f}美元
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📈 近期成交價格直方圖")
        
        fig, ax = plt.subplots(figsize=(7, 4.2))
        
        counts, bins, patches = ax.hist(prices, bins=10, color='cornflowerblue', edgecolor='black', alpha=0.8, label='成交次數 (頻率)')
        
        kde = gaussian_kde(prices)
        x_grid = np.linspace(min_price - 20, max_price + 20, 200)
        bin_width = bins[1] - bins[0]
        kde_scaled = kde(x_grid) * len(prices) * bin_width
        ax.plot(x_grid, kde_scaled, color='red', linewidth=2.5, label='趨勢擬合線')
        
        ax.axvspan(p15, p85, color='gold', alpha=0.25)
        ax.text((p15+p85)/2, max(counts)*0.05, "70% 主要買價區間", ha='center', va='bottom', color='dimgray', fontsize=9, fontweight='bold')
        
        ax.axvline(ask_price, color='dodgerblue', linewidth=2)
        ax.text(ask_price, max(counts)*1.02, f"現買價\n${ask_price:.0f}", color='dodgerblue', ha='center', fontweight='bold', fontsize=9)
        
        ax.axvline(p25, color='green', linestyle='--', linewidth=1.5)
        ax.text(p25, max(counts)*1.02, f"理想買價\n${p25:.0f}", color='green', ha='center', fontsize=9)
        
        ax.axvline(mean_price, color='purple', linestyle='--', linewidth=1.5)
        ax.text(mean_price, max(counts)*1.1, f"平均價\n${mean_price:.0f}", color='purple', ha='center', fontsize=9)
        
        ax.axvline(p75, color='red', linestyle='--', linewidth=1.5)
        ax.text(p75, max(counts)*1.02, f"溢買價\n${p75:.0f}", color='red', ha='center', fontsize=9)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.set_xlabel("成交金額 (USD) ➔ 越往右代表買得越貴", fontsize=10, color='gray')
        ax.set_ylabel("頻率 (成交筆數)", fontsize=10, color='gray')
        ax.legend(loc='upper right', frameon=False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    # ----------------- 右側 -----------------
    with col_right:
        st.markdown("### 🎯 目前四大價格明細表")
        df_table = pd.DataFrame({
            "價格類型": ["現買價 (Ask)", "理想買價 (P25)", "溢買價格 (P75)", "70% 主要買價區間"],
            "金額 (USD)": [f"US$ {ask_price:.1f}", f"US$ {p25:.1f}", f"US$ {p75:.1f}", f"US$ {p15:.1f} - {p85:.1f}"],
            "分布位置與解讀": [
                "當前賣家掛單價，低於理想買價即可秒殺", 
                "前 25% 低價區，最具 CP 值", 
                "超過 75% 歷史價，屬高價追高區", 
                "涵蓋歷史 70% 的集中成交區間"
            ]
        })
        st.markdown(df_table.to_html(index=False, classes="table table-striped", justify="left"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        if ask_price <= p25:
            rec_title = "🔥 極限推薦秒殺"
            rec_desc = f"目前賣家掛單價 (${ask_price:.0f}) 已低於理想買價P25 (${p25:.0f})，位於歷史低價區，建議直接點擊購買秒殺！"
            border_color = "#2ca02c"
        elif ask_price <= p50:
            rec_title = "✅ 合理市價區間"
            rec_desc = f"目前掛單價 (${ask_price:.0f}) 落在合理中位數 (${p50:.0f}) 以內，價格公道，可考慮入手。"
            border_color = "#ff7f0e"
        else:
            rec_title = "⚠️ 偏高需觀望 (FOMO 警告)"
            rec_desc = f"目前掛單價 (${ask_price:.0f}) 偏高，處於歷史高價區間，建議掛較低的 Bid 等待，不要衝動追高！"
            border_color = "#d62728"
            
        st.markdown(f"""
        <div style="border-left: 6px solid {border_color}; background-color: #F8F9FA; padding: 15px; border-radius: 5px; margin-top: 15px;">
            <div style="font-weight: bold; font-size: 16px; color: {border_color}; margin-bottom: 5px;">
                💡 買家出價策略建議 : {rec_title}
            </div>
            <div style="font-size: 14px; color: #333;">{rec_desc}</div>
            <div style="font-size: 12px; color: gray; margin-top: 8px;">
                (資料參考：建議出價屬於百分補P25 (${p25:.0f}) 為核心基準)
            </div>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"執行錯誤：{e}")
