import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from vnstock import *

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Tối Ưu Đa Ngành Tinh Chỉnh", layout="wide", page_icon="🚀")
st.title("🚀 Hệ Thống Tối Ưu Tín Hiệu Dòng Tiền & Động Lượng Tinh Chỉnh (Alpha Max)")
st.caption("Thuật toán đã được tinh chỉnh: Sử dụng Động lượng liên tục (Percentile Rank) và Bộ lọc xu hướng kép MA20/MA50 nhằm đánh bại VN-Index.")

# ==========================================
# 2. TỰ ĐỘNG TẢI VÀ PHÂN LOẠI NGÀNH ĐỘNG
# ==========================================
@st.cache_data(ttl=86400)
def get_dynamic_sector_map():
    try:
        df_comp = listing_companies()
        if df_comp is not None and not df_comp.empty:
            col_ticker = 'ticker' if 'ticker' in df_comp.columns else ('symbol' if 'symbol' in df_comp.columns else df_comp.columns[0])
            col_group = 'comGroupCode' if 'comGroupCode' in df_comp.columns else ('exchange' if 'exchange' in df_comp.columns else None)
            col_industry = None
            for c in ['industryName', 'industry', 'ngành', 'nganh']:
                if c in df_comp.columns: col_industry = c; break
            
            if col_industry is None: raise KeyError("Không tìm thấy cột ngành.")
            df_filtered = df_comp[(df_comp[col_group] == 'HOSE') & (df_comp[col_ticker].str.len() == 3) & (df_comp[col_industry].notna())].copy()
            
            dynamic_map = {}
            for sector_name, group in df_filtered.groupby(col_industry):
                tickers = group[col_ticker].str.strip().unique().tolist()
                if len(tickers) >= 2: dynamic_map[sector_name.strip()] = tickers
            return dynamic_map
    except: pass
    return {
        "Ngân hàng": ['VCB', 'BID', 'CTG', 'TCB', 'MBB', 'VPB', 'ACB', 'STB', 'HDB', 'TPB'],
        "Bất động sản": ['VHM', 'VIC', 'VRE', 'NVL', 'PDR', 'KDH', 'NLG', 'DXG', 'DIG'],
        "Thép & Vật liệu": ['HPG', 'HSG', 'NKG'],
        "Dịch vụ tài chính (Chứng khoán)": ['SSI', 'VND', 'VCI', 'HCM', 'FTS', 'BSI'],
        "Bán lẻ - Công nghệ - Tiêu dùng": ['FPT', 'MWG', 'MSN', 'VNM', 'FRT', 'DGW']
    }

DYNAMIC_SECTOR_MAP = get_dynamic_sector_map()
TICKER_TO_SECTOR = {t: s for s, t_list in DYNAMIC_SECTOR_MAP.items() for t in t_list}

# ==========================================
# 3. THANH ĐIỀU HƯỚNG CẤU HÌNH (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Cấu Hình Tinh Chỉnh")

all_available_sectors = list(DYNAMIC_SECTOR_MAP.keys())
selected_sectors = st.sidebar.multiselect("Chọn các nhóm ngành quét:", options=all_available_sectors, default=all_available_sectors[:5])

final_scan_tickers = []
for s in selected_sectors: final_scan_tickers.extend(DYNAMIC_SECTOR_MAP[s])
final_scan_tickers = list(set(final_scan_tickers))

portfolio_size = st.sidebar.slider("Số lượng mã trong danh mục (N)", min_value=5, max_value=10, value=5, step=1)
max_stocks_per_sector = st.sidebar.slider("Số mã tối đa/ngành (Kiểm soát đa dạng hóa)", min_value=1, max_value=4, value=2)

rf_annual = st.sidebar.number_input("Lãi suất phi rủi ro (RF)", value=0.045, step=0.005)
trading_days = st.sidebar.number_input("Số ngày giao dịch/năm", value=252)

st.sidebar.subheader("📅 Khung Thời Gian Tách Biệt")
train_start = st.sidebar.date_input("Huấn luyện từ ngày", datetime(2022, 1, 1))
train_end = st.sidebar.date_input("Huấn luyện đến ngày", datetime(2024, 12, 31))
test_start = st.sidebar.date_input("Kiểm thử từ ngày", datetime(2025, 1, 1))
test_end = st.sidebar.date_input("Kiểm thử đến ngày", datetime(2025, 12, 31))

# ==========================================
# 4. HÀM TẢI VÀ XỬ LÝ DỮ LIỆU
# ==========================================
@st.cache_data(ttl=3600)
def load_all_periods_data(ticker_list, t_start, t_end):
    start_str, end_str = t_start.strftime('%Y-%m-%d'), t_end.strftime('%Y-%m-%d')
    series_dict = {}
    for ticker in ticker_list:
        try:
            df = stock_historical_data(symbol=ticker, start_date=start_str, end_date=end_str, resolution='1D', type='stock')
            if df is not None and not df.empty:
                d_col = 'time' if 'time' in df.columns else 'date'
                df[d_col] = pd.to_datetime(df[d_col])
                series_dict[ticker] = df.drop_duplicates(subset=[d_col]).set_index(d_col).sort_index()['close']
        except: continue
    try:
        df_idx = stock_historical_data(symbol='VNINDEX', start_date=start_str, end_date=end_str, resolution='1D', type='index')
        if df_idx is not None and not df_idx.empty:
            d_col = 'time' if 'time' in df_idx.columns else 'date'
            df_idx[d_col] = pd.to_datetime(df_idx[d_col])
            series_dict['VNINDEX'] = df_idx.drop_duplicates(subset=[d_col]).set_index(d_col).sort_index()['close']
    except: pass
    return pd.concat(series_dict, axis=1).sort_index().ffill().bfill() if series_dict else pd.DataFrame()

# ==========================================
# 5. KÍCH HOẠT TÍNH TOÁN
# ==========================================
if st.sidebar.button("🚀 Thực Thi Thuật Toán Tinh Chỉnh", type="primary"):
    if train_end >= test_start:
        st.error("❌ Ngày kết thúc huấn luyện phải trước Ngày bắt đầu kiểm thử!")
    else:
        with st.spinner("Đang trích xuất ma trận Động lượng liên tục..."):
            df_train = load_all_periods_data(final_scan_tickers, train_start, train_end)
            
            if df_train.empty:
                st.error("❌ Không tải được dữ liệu.")
            else:
                raw_metrics = []
                stock_cols = [c for c in df_train.columns if c != 'VNINDEX']
                
                # Tính toán các chỉ báo thô cho toàn sàn
                for ticker in stock_cols:
                    series = df_train[ticker].dropna()
                    if len(series) < 60: continue
                    
                    current_price = series.iloc[-1]
                    ma_20 = series.rolling(window=20).mean().iloc[-1]
                    ma_50 = series.rolling(window=50).mean().iloc[-1]
                    
                    # TINH CHỈNH 2: Bộ lọc xu hướng kép nghiêm ngặt
                    trend_valid = 1 if (current_price > ma_20 and current_price > ma_50) else 0
                    
                    # RSI
                    delta = series.diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean().iloc[-1]
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean().iloc[-1]
                    rsi = 100 - (100 / (1 + (gain / (loss + 1e-12))))
                    rsi_score = 1.0 if (45 <= rsi <= 75) else 0.2
                    
                    # TINH CHỈNH 1: Lấy giá trị Động lượng liên tục thô để xếp hạng phần trăm
                    momentum_1m_raw = (series.iloc[-1] / series.iloc[-20]) - 1 if len(series) >= 20 else -0.99
                    
                    raw_metrics.append({
                        "Ticker": ticker, "Sector": TICKER_TO_SECTOR.get(ticker, "Khác"),
                        "Price": current_price, "Trend_Valid": trend_valid,
                        "RSI": rsi, "RSI_Score": rsi_score, "Mom_Raw": momentum_1m_raw
                    })
                
                df_raw = pd.DataFrame(raw_metrics)
                
                if df_raw.empty:
                    st.error("❌ Không có đủ dữ liệu cổ phiếu hợp lệ.")
                else:
                    # Chuyển đổi Động lượng thô thành Điểm thứ hạng phần trăm (Percentile Rank từ 0 đến 1)
                    df_raw["Mom_Score"] = df_raw["Mom_Raw"].rank(pct=True)
                    
                    # Tính điểm tổng hợp tinh chỉnh
                    df_raw["Total_Score"] = (df_raw["Trend_Valid"] * 0.4) + (df_raw["Mom_Score"] * 0.4) + (df_raw["RSI_Score"] * 0.2)
                    df_raw = df_raw.sort_values(by="Total_Score", ascending=False)
                    
                    # Sàng lọc danh mục kết hợp chặn trần rủi ro ngành
                    portfolio_list = []
                    sector_counts = {}
                    
                    for idx, row in df_raw.iterrows():
                        if row["Trend_Valid"] != 1: continue # Chỉ chọn cổ phiếu Uptrend mạnh
                        
                        sec = row["Sector"]
                        count = sector_counts.get(sec, 0)
                        if count < max_stocks_per_sector:
                            portfolio_list.append({
                                "Mã Cổ Phiếu": row["Ticker"], "Nhóm Ngành": sec,
                                "Giá Chốt HL": row["Price"], "Chỉ Số RSI": round(row["RSI"], 1),
                                "Hiệu Suất 1M": f"{row['Mom_Raw']*100:.1f}%", "Điểm Tối Ưu": round(row["Total_Score"], 3)
                            })
                            sector_counts[sec] = count + 1
                        if len(portfolio_list) == portfolio_size: break
                    
                    st.subheader(f"🎯 Danh Mục {len(portfolio_list)} Siêu Cổ Phiếu Dòng Tiền Được Chọn")
                    df_port = pd.DataFrame(portfolio_list)
                    st.dataframe(df_port, use_container_width=True, hide_index=True)
                    
                    # --- TINH CHỈNH 3: PHÂN BỔ VỐN 80-20 ĐỘNG THEO Pareto NÂNG CAO ---
                    sorted_tickers = df_port["Mã Cổ Phiếu"].tolist()
                    sorted_scores = df_port["Điểm Tối Ưu"].tolist()
                    n_assets = len(sorted_tickers)
                    
                    weights = np.zeros(n_assets)
                    if n_assets >= 2:
                        # 80% vốn dồn cho Top 2 mã, chia theo tỷ lệ điểm số tương đối của chúng để tối ưu Alpha
                        sum_top2_score = sorted_scores[0] + sorted_scores[1]
                        weights[0] = 0.80 * (sorted_scores[0] / sum_top2_score)
                        weights[1] = 0.80 * (sorted_scores[1] / sum_top2_score)
                        
                        # 20% vốn còn lại chia đều cho các mã vệ tinh phía sau
                        if n_assets > 2:
                            rem_w = 0.20 / (n_assets - 2)
                            for j in range(2, n_assets): weights[j] = rem_w
                    else:
                        weights[0] = 1.0
                    
                    df_w = pd.DataFrame({"Cổ Phiếu": sorted_tickers, "Ngành": df_port["Nhóm Ngành"], "Tỷ Trọng Vốn": [f"{w*100:.2f}%" for w in weights]})
                    st.markdown("##### 💰 Cơ Cấu Phân Bổ Vốn Pareto Động Đã Tối Ưu:")
                    st.dataframe(df_w.T, use_container_width=True)
                    
                    # --- BACKTEST OUT-OF-SAMPLE (GIAI ĐOẠN 2025 THỰC TẾ) ---
                    with st.spinner("Đang chạy Backtest Out-of-Sample..."):
                        df_test = load_all_periods_data(final_scan_tickers, test_start, test_end)
                        
                        if df_test.empty or 'VNINDEX' not in df_test.columns:
                            st.error("❌ Không lấy được dữ liệu tập kiểm thử năm 2025.")
                        else:
                            df_test_ret = df_test.pct_change().dropna()
                            
                            # Tính toán lợi nhuận tích lũy
                            p_ret = df_test_ret[sorted_tickers].dot(weights)
                            cum_p = (1 + p_ret).cumprod()
                            cum_m = (1 + df_test_ret['VNINDEX']).cumprod()
                            
                            valid_all = [t for t in final_scan_tickers if t in df_test_ret.columns]
                            b_ret = df_test_ret[valid_all].dot(np.ones(len(valid_all)) / len(valid_all))
                            cum_b = (1 + b_ret).cumprod()
                            
                            # Thống kê định lượng
                            p_ann = p_ret.mean() * trading_days
                            m_ann = df_test_ret['VNINDEX'].mean() * trading_days
                            b_ann = b_ret.mean() * trading_days
                            
                            p_sd = p_ret.std() * np.sqrt(trading_days)
                            m_sd = df_test_ret['VNINDEX'].std() * np.sqrt(trading_days)
                            b_sd = b_ret.std() * np.sqrt(trading_days)
                            
                            p_sh = (p_ann - rf_annual) / p_sd if p_sd != 0 else 0
                            m_sh = (m_ann - rf_annual) / m_sd if m_sd != 0 else 0
                            b_sh = (b_ann - rf_annual) / b_sd if b_sd != 0 else 0
                            
                            beta = p_ret.cov(df_test_ret['VNINDEX']) / df_test_ret['VNINDEX'].var()
                            alpha = p_ann - (rf_annual + beta * (m_ann - rf_annual))
                            
                            # Hiển thị Ma trận kết quả
                            st.subheader("📐 Ma Trận Đánh Giá Hiệu Quả Sau Khi Tinh Chỉnh Thuật Toán")
                            df_metrics = pd.DataFrame({
                                "Chỉ tiêu định lượng": ["Lợi nhuận TB năm", "Độ biến động rủi ro", "Hệ số Sharpe Ratio"],
                                "DANH MỤC TINH CHỈNH (ALPHA MAX)": [f"{p_ann*100:.2f}%", f"{p_sd*100:.2f}%", f"{p_sh:.4f}"],
                                "Chiến lược cơ sở (Buy & Hold)": [f"{b_ann*100:.2f}%", f"{b_sd*100:.2f}%", f"{b_sh:.4f}"],
                                "Thị Trường Chung (VN-Index)": [f"{m_ann*100:.2f}%", f"{m_sd*100:.2f}%", f"{m_sh:.4f}"]
                            })
                            
                            c1, c2 = st.columns([3, 2])
                            with c1: st.dataframe(df_metrics, use_container_width=True, hide_index=True)
                            with c2:
                                st.metric("Chỉ số Thặng dư Alpha thực tế", f"{alpha*100:.2f}%", help="Mức vượt trội tuyệt đối của chiến lược so với kỳ vọng rủi ro.")
                                st.metric("Hệ số biến động Beta", f"{beta:.2f}")
                            
                            # Biểu đồ tài sản
                            fig, ax = plt.subplots(figsize=(12, 5))
                            ax.plot(cum_p.index, cum_p.values, color='crimson', lw=2.5, label="Danh Mục Tinh Chỉnh (Alpha Max)")
                            ax.plot(cum_b.index, cum_b.values, color='orange', lw=1.5, linestyle='--', label="Chiến Lược Cơ Sở")
                            ax.plot(cum_m.index, cum_m.values, color='black', lw=1.5, alpha=0.6, label="Thị Trường Chung (VN-Index)")
                            ax.set_title("So Sánh Tài Sản Tích Lũy Sau Tinh Chỉnh (Out-of-Sample)")
                            ax.grid(True, linestyle=':')
                            ax.legend()
                            st.pyplot(fig)
