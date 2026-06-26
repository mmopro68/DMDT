import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from vnstock import *

# ==========================================
# CẤU HÌNH GIAO DIỆN CHUYÊN NGHIỆP
# ==========================================
st.set_page_config(page_title="Alpha Max Portfolio System", layout="wide", page_icon="📈")

# Tối ưu giao diện bằng CSS tùy chỉnh
st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
    .stMetric {background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 5px solid #007bff;}
    .report-card {background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 15px;}
    </style>
""", unsafe_allow_html=True)

st.title("📈 Hệ Thống Phân Tích Định Lượng & Tối Ưu Hóa Danh Mục Cao Cấp (Alpha Max)")
st.caption("Phiên bản thuật toán tinh chỉnh nâng cao: Tích hợp Bộ lọc Xu hướng đa tầng và Động lượng điều chỉnh rủi ro nhằm chinh phục VN-Index.")

# ==========================================
# TỰ ĐỘNG TẢI VÀ PHÂN LOẠI NGÀNH ĐỘNG
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
# BỐ CỤC 1 ĐẾN 8: BỘ THAM SỐ ĐẦU VÀO TRÊN SIDEBAR
# ==========================================
st.sidebar.header("⚙️ BẢNG CẤU HÌNH THAM SỐ")

# 1. Lựa chọn nhóm ngành
all_available_sectors = list(DYNAMIC_SECTOR_MAP.keys())
selected_sectors = st.sidebar.multiselect(
    "1. Lựa chọn nhóm ngành quét vốn:", 
    options=all_available_sectors, 
    default=all_available_sectors[:5]
)

final_scan_tickers = []
for s in selected_sectors: final_scan_tickers.extend(DYNAMIC_SECTOR_MAP[s])
final_scan_tickers = list(set(final_scan_tickers))

# 2 & 3. Định biên số lượng mã danh mục
portfolio_size = st.sidebar.slider("2. Số lượng mã trong danh mục (N)", min_value=5, max_value=10, value=5, step=1)
max_stocks_per_sector = st.sidebar.slider("3. Số lượng mã tối đa/ngành", min_value=1, max_value=4, value=2)

# 4 & 5. Định biên thông số thị trường
rf_annual = st.sidebar.number_input("4. Lãi suất phi rủi ro (RF)", value=0.045, step=0.005, format="%.3f")
trading_days = st.sidebar.number_input("5. Số ngày giao dịch/năm", value=252, step=1)

# 6 & 7. Khung thời gian tách biệt
st.sidebar.subheader("📅 KHUNG THỜI GIAN KIỂM ĐỊNH")
train_start = st.sidebar.date_input("6. Ngày bắt đầu huấn luyện", datetime(2022, 1, 1))
train_end = st.sidebar.date_input("6. Ngày kết thúc huấn luyện", datetime(2024, 12, 31))
test_start = st.sidebar.date_input("7. Ngày bắt đầu backtest", datetime(2025, 1, 1))
test_end = st.sidebar.date_input("7. Ngày kết thúc backtest", datetime(2025, 12, 31))

# 8. Lựa chọn phương án phân bổ vốn
st.sidebar.subheader("🎯 CHIẾN LƯỢC QUẢN TRỊ VỐN")
strategy_option = st.sidebar.radio(
    "8. Lựa chọn phương án phân bổ vốn:", 
    [
        "Phân bổ đều (Equal Weight)", 
        "Phân bổ theo trọng số điểm tín hiệu (Score Weight)",
        "Chiến lược 80-20 (80% vốn tập trung dồn cho Top 2 mã mạnh nhất)"
    ]
)

# ==========================================
# HÀM TẢI VÀ XỬ LÝ DỮ LIỆU GỐC
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

def calculate_max_drawdown(cum_returns_series):
    rolling_max = cum_returns_series.cummax()
    drawdowns = (cum_returns_series - rolling_max) / rolling_max
    return drawdowns.min()

# ==========================================
# LOGIC THỰC THI CHÍNH
# ==========================================
if st.sidebar.button("🚀 KÍCH HOẠT HỆ THỐNG ALPHA MAX", type="primary"):
    if train_end >= test_start:
        st.error("❌ Lỗi: Ngày kết thúc huấn luyện phải trước Ngày bắt đầu backtest để đảm bảo tính khách quan!")
    else:
        with st.spinner("Hệ thống đang quét toàn thị trường và tối ưu hóa ma trận Alpha..."):
            df_train = load_all_periods_data(final_scan_tickers, train_start, train_end)
            
            if df_train.empty:
                st.error("❌ Không trích xuất được dữ liệu giao dịch lịch sử.")
            else:
                raw_metrics = []
                stock_cols = [c for c in df_train.columns if c != 'VNINDEX']
                
                for ticker in stock_cols:
                    series = df_train[ticker].dropna()
                    if len(series) < 200: continue # Đảm bảo đủ dữ liệu tính toán MA200
                    
                    current_price = series.iloc[-1]
                    ma_20 = series.rolling(window=20).mean().iloc[-1]
                    ma_50 = series.rolling(window=50).mean().iloc[-1]
                    ma_200 = series.rolling(window=200).mean().iloc[-1]
                    
                    # TINH CHỈNH THUẬT TOÁN CỐT LÕI: Bộ lọc xu hướng dài hạn đa tầng MA200
                    trend_valid = 1 if (current_price > ma_20 and current_price > ma_50 and current_price > ma_200) else 0
                    
                    # Tính toán RSI
                    delta = series.diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean().iloc[-1]
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean().iloc[-1]
                    rsi = 100 - (100 / (1 + (gain / (loss + 1e-12))))
                    rsi_score = 1.0 if (50 <= rsi <= 70) else 0.1 # Thu hẹp vùng an toàn để chọn mã khỏe hẳn
                    
                    # TINH CHỈNH THUẬT TOÁN ĐỘNG LƯỢNG: Động lượng điều chỉnh rủi ro (Risk-Adjusted Momentum)
                    returns_1m = series.pct_change(20).iloc[-1]
                    volatility_1m = series.pct_change().tail(20).std() + 1e-12
                    risk_adj_mom = returns_1m / volatility_1m
                    
                    raw_metrics.append({
                        "Ticker": ticker, "Sector": TICKER_TO_SECTOR.get(ticker, "Khác"),
                        "Price": current_price, "Trend_Valid": trend_valid, "RSI": rsi,
                        "RSI_Score": rsi_score, "Mom_Raw": risk_adj_mom
                    })
                
                df_raw = pd.DataFrame(raw_metrics)
                
                if df_raw.empty:
                    st.error("❌ Không tìm thấy cổ phiếu nào vượt qua bộ lọc sơ bộ.")
                else:
                    # Chấm điểm xếp hạng động lượng phần trăm toàn thị trường
                    df_raw["Mom_Score"] = df_raw["Mom_Raw"].rank(pct=True)
                    df_raw["Total_Score"] = (df_raw["Trend_Valid"] * 0.4) + (df_raw["Mom_Score"] * 0.4) + (df_raw["RSI_Score"] * 0.2)
                    df_raw = df_raw.sort_values(by="Total_Score", ascending=False)
                    
                    # Sàng lọc danh mục theo tiêu chí khắt khe và chống rủi ro hệ thống ngành
                    portfolio_list = []
                    sector_counts = {}
                    
                    for idx, row in df_raw.iterrows():
                        if row["Trend_Valid"] != 1: continue # Loại bỏ thẳng tay các mã gãy xu hướng dài hạn
                        
                        sec = row["Sector"]
                        count = sector_counts.get(sec, 0)
                        if count < max_stocks_per_sector:
                            portfolio_list.append({
                                "Hạng": len(portfolio_list) + 1,
                                "Mã Cổ Phiếu": row["Ticker"], "Nhóm Ngành Lĩnh Vực": sec,
                                "Giá Chốt HL": row["Price"], "Chỉ Số RSI": round(row["RSI"], 1),
                                "Điểm Đánh Giá Tín Hiệu": round(row["Total_Score"], 3)
                            })
                            sector_counts[sec] = count + 1
                        if len(portfolio_list) == portfolio_size: break
                    
                    # 9. DANH MỤC CỔ PHIẾU ĐƯỢC CHỌN THEO BỘ LỌC TÍN HIỆU
                    st.markdown('<div class="report-card">', unsafe_allow_html=True)
                    st.subheader("📋 9. Danh mục cổ phiếu được chọn theo bộ lọc tín hiệu Alpha Max")
                    if not portfolio_list:
                        st.error("❌ Không có cổ phiếu nào thỏa mãn bộ lọc xu hướng dài hạn an toàn.")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        df_port = pd.DataFrame(portfolio_list)
                        st.dataframe(df_port, use_container_width=True, hide_index=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        sorted_tickers = df_port["Mã Cổ Phiếu"].tolist()
                        sorted_scores = df_port["Điểm Đánh Giá Tín Hiệu"].tolist()
                        n_assets = len(sorted_tickers)
                        
                        # 10. CƠ CẤU PHÂN BỔ VỐN
                        st.markdown('<div class="report-card">', unsafe_allow_html=True)
                        st.subheader("💰 10. Cơ cấu phân bổ tỷ trọng dòng vốn")
                        
                        weights = np.zeros(n_assets)
                        if strategy_option == "Phân bổ đều (Equal Weight)":
                            weights = np.ones(n_assets) / n_assets
                        elif strategy_option == "Phân bổ theo trọng số điểm tín hiệu (Score Weight)":
                            total_s = sum(sorted_scores)
                            weights = np.array(sorted_scores) / total_s if total_s > 0 else np.ones(n_assets) / n_assets
                        elif strategy_option == "Chiến lược 80-20 (80% vốn tập trung dồn cho Top 2 mã mạnh nhất)":
                            if n_assets >= 2:
                                # Tinh chỉnh: Chia 80% vốn động theo tỷ lệ điểm số tương đối của Top 2 để dồn lực cho mã mạnh nhất
                                sum_top2 = sorted_scores[0] + sorted_scores[1]
                                weights[0] = 0.80 * (sorted_scores[0] / sum_top2)
                                weights[1] = 0.80 * (sorted_scores[1] / sum_top2)
                                if n_assets > 2:
                                    rem_w = 0.20 / (n_assets - 2)
                                    for j in range(2, n_assets): weights[j] = rem_w
                            else: weights[0] = 1.0
                        
                        df_w_display = pd.DataFrame({
                            "Mã": sorted_tickers, "Ngành": df_port["Nhóm Ngành Lĩnh Vực"],
                            "Tỷ Trọng Phân Bổ Vốn": [f"{w*100:.2f}%" for w in weights]
                        })
                        st.dataframe(df_w_display.T, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # 11. ĐÁNH GIÁ HIỆU QUẢ DANH MỤC (BACKTEST OUT-OF-SAMPLE)
                        st.markdown('<div class="report-card">', unsafe_allow_html=True)
                        st.subheader("📊 11. Đánh giá hiệu quả danh mục thực tế (Out-of-Sample Backtest 2025)")
                        
                        df_test = load_all_periods_data(final_scan_tickers, test_start, test_end)
                        
                        if df_test.empty or 'VNINDEX' not in df_test.columns:
                            st.error("❌ Không trích xuất được dữ liệu kiểm thử thực tế của năm 2025.")
                        else:
                            df_test_ret = df_test.pct_change().dropna()
                            
                            # Tính chuỗi lợi nhuận thực tế
                            portfolio_ret = df_test_ret[sorted_tickers].dot(weights)
                            cum_portfolio = (1 + portfolio_ret).cumprod()
                            
                            vnindex_ret = df_test_ret['VNINDEX']
                            cum_vnindex = (1 + vnindex_ret).cumprod()
                            
                            valid_all = [t for t in final_scan_tickers if t in df_test_ret.columns]
                            base_weights = np.ones(len(valid_all)) / len(valid_all)
                            base_ret = df_test_ret[valid_all].dot(base_weights)
                            cum_base = (1 + base_ret).cumprod()
                            
                            # Tính toán các chỉ tiêu định lượng năm
                            p_ann = portfolio_ret.mean() * trading_days
                            m_ann = vnindex_ret.mean() * trading_days
                            b_ann = base_ret.mean() * trading_days
                            
                            p_sd = portfolio_ret.std() * np.sqrt(trading_days)
                            m_sd = vnindex_ret.std() * np.sqrt(trading_days)
                            b_sd = base_ret.std() * np.sqrt(trading_days)
                            
                            p_sharpe = (p_ann - rf_annual) / (p_sd + 1e-12)
                            m_sharpe = (m_ann - rf_annual) / (m_sd + 1e-12)
                            b_sharpe = (b_ann - rf_annual) / (b_sd + 1e-12)
                            
                            p_max_dd = calculate_max_drawdown(cum_portfolio)
                            m_max_dd = calculate_max_drawdown(cum_vnindex)
                            b_max_dd = calculate_max_drawdown(cum_base)
                            
                            # Tính hệ số Alpha và Beta theo CAPM
                            cov_m = portfolio_ret.cov(vnindex_ret)
                            var_m = vnindex_ret.var()
                            beta_val = cov_m / var_m if var_m != 0 else 1.0
                            alpha_jensen = p_ann - (rf_annual + beta_val * (m_ann - rf_annual))
                            
                            # Hiển thị Metrics dạng khối chuyên nghiệp
                            c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                            with c_m1: st.metric("Lợi nhuận Danh mục Tinh chỉnh", f"{p_ann*100:.2f}%", delta=f"{(p_ann - m_ann)*100:.2f}% vs Index")
                            with c_m2: st.metric("Chỉ số Sharpe Ratio (Chiến lược)", f"{p_sharpe:.4f}")
                            with c_m3: st.metric("Alpha Jensen Thặng dư", f"{alpha_jensen*100:.2f}%")
                            with c_m4: st.metric("Mức sụt giảm lớn nhất (Max DD)", f"{p_max_dd*100:.2f}%")
                            
                            # Bảng so sánh chi tiết giữa 3 phương án
                            df_metrics_final = pd.DataFrame({
                                "Chỉ tiêu kiểm thử thực nghiệm (2025)": ["Lợi nhuận TB năm ($E_R$)", "Độ rủi ro biến động ($\sigma$)", "Hệ số Sharpe", "Sụt giảm tài sản tối đa (Max DD)"],
                                "DANH MỤC TINH CHỈNH ĐẠT ALPHA": [f"{p_ann*100:.2f}%", f"{p_sd*100:.2f}%", f"{p_sharpe:.4f}", f"{p_max_dd*100:.2f}%"],
                                "Chiến lược cơ sở (Buy & Hold Toàn rổ)": [f"{b_ann*100:.2f}%", f"{b_sd*100:.2f}%", f"{b_sharpe:.4f}", f"{b_max_dd*100:.2f}%"],
                                "Thị Trường Chung (VN-Index)": [f"{m_ann*100:.2f}%", f"{m_sd*100:.2f}%", f"{m_sharpe:.4f}", f"{m_max_dd*100:.2f}%"]
                            })
                            st.markdown("---")
                            st.dataframe(df_metrics_final, use_container_width=True, hide_index=True)
                            
                            # Biểu đồ tăng trưởng tài sản
                            st.markdown("##### 📉 Biểu đồ tăng trưởng tài sản tích lũy thực tế công khai (Out-of-Sample Performance)")
                            fig, ax = plt.subplots(figsize=(12, 5))
                            ax.plot(cum_portfolio.index, cum_portfolio.values, color='#008080', lw=2.5, label="Chiến lược Tinh chỉnh Alpha Max (Đã tối ưu)")
                            ax.plot(cum_base.index, cum_base.values, color='#FFA500', lw=1.5, linestyle='--', label="Chiến lược Cơ sở")
                            ax.plot(cum_vnindex.index, cum_vnindex.values, color='#2F4F4F', lw=1.5, alpha=0.5, label="Thị Trường Chung (VN-Index)")
                            ax.set_ylabel("Giá trị tài sản tích lũy (Gốc = 1.0)")
                            ax.grid(True, linestyle=':')
                            ax.legend()
                            st.pyplot(fig)
                        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("💡 Hệ thống đã được nâng cấp sang cấu trúc thuật toán Alpha Max thế hệ mới. Vui lòng thiết lập cấu hình tham số đầu vào bên trái và bấm nút Khởi chạy.")
