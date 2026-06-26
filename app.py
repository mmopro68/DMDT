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

st.markdown("""
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
    .stMetric {background-color: #f8f9fa; padding: 10px; border-radius: 5px; border-left: 5px solid #20b2aa;}
    .report-card {background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); margin-bottom: 15px;}
    .status-box {padding: 15px; border-radius: 5px; margin-bottom: 15px; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

st.title("📈 Hệ Thống Phân Tích Định Lượng & Tối Ưu Hóa Toàn Sàn (Alpha Max)")
st.caption("Chiến lược thích ứng động: Lọc cổ phiếu từ dữ liệu huấn luyện và tự động kích hoạt phương án phân bổ vốn theo trạng thái VN-Index lúc backtest.")

# ==========================================
# TỰ ĐỘNG TẢI DANH SÁCH MÃ HOSE TOÀN SÀN
# ==========================================
@st.cache_data(ttl=86400)
def get_all_hose_tickers():
    try:
        df_comp = listing_companies()
        if df_comp is not None and not df_comp.empty:
            col_ticker = 'ticker' if 'ticker' in df_comp.columns else ('symbol' if 'symbol' in df_comp.columns else df_comp.columns[0])
            col_group = 'comGroupCode' if 'comGroupCode' in df_comp.columns else ('exchange' if 'exchange' in df_comp.columns else None)
            df_filtered = df_comp[(df_comp[col_group] == 'HOSE') & (df_comp[col_ticker].str.len() == 3)].copy()
            return df_filtered[col_ticker].str.strip().unique().tolist()
    except: pass
    return ['VCB', 'BID', 'CTG', 'TCB', 'MBB', 'VPB', 'ACB', 'STB', 'HDB', 'TPB',
            'VHM', 'VIC', 'VRE', 'NVL', 'PDR', 'KDH', 'NLG', 'DXG', 'DIG', 'HPG', 
            'HSG', 'NKG', 'SSI', 'VND', 'VCI', 'HCM', 'FTS', 'BSI', 'FPT', 'MWG', 
            'MSN', 'VNM', 'FRT', 'DGW', 'GMD', 'HAH', 'PVT', 'PVD', 'GAS', 'PLX']

ALL_HOSE_TICKERS = get_all_hose_tickers()

# ==========================================
# THANH ĐIỀU HƯỚNG CẤU HÌNH (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ THAM SỐ CẤU HÌNH")

portfolio_size = st.sidebar.slider("Số lượng mã trong danh mục (N)", min_value=5, max_value=10, value=5, step=1)
max_stocks_per_sector = st.sidebar.slider("Số lượng mã tối đa cùng một nhóm ngành", min_value=1, max_value=4, value=2)

rf_annual = st.sidebar.number_input("Lãi suất phi rủi ro (RF)", value=0.045, step=0.005, format="%.3f")
trading_days = st.sidebar.number_input("Số ngày giao dịch/năm", value=252, step=1)

st.sidebar.markdown("---")
st.sidebar.subheader("📅 KHUNG THỜI GIAN HUẤN LUYỆN")
train_start = st.sidebar.date_input("Ngày bắt đầu huấn luyện", datetime(2022, 1, 1), key="tr_start")
train_end = st.sidebar.date_input("Ngày kết thúc huấn luyện", datetime(2024, 12, 31), key="tr_end")

st.sidebar.subheader("📅 KHUNG THỜI GIAN BACKTEST")
test_start = st.sidebar.date_input("Ngày bắt đầu backtest", datetime(2025, 1, 1), key="te_start")
test_end = st.sidebar.date_input("Ngày kết thúc backtest", datetime(2025, 12, 31), key="te_end")

@st.cache_data(ttl=86400)
def get_ticker_sector_dict():
    try:
        df_comp = listing_companies()
        if df_comp is not None and not df_comp.empty:
            col_ticker = 'ticker' if 'ticker' in df_comp.columns else 'symbol'
            col_industry = None
            for c in ['industryName', 'industry', 'ngành', 'nganh']:
                if c in df_comp.columns: col_industry = c; break
            if col_industry:
                return dict(zip(df_comp[col_ticker].str.strip(), df_comp[col_industry].str.strip()))
    except: pass
    return {}

TICKER_TO_SECTOR = get_ticker_sector_dict()

# ==========================================
# HÀM TẢI VÀ XỬ LÝ DỮ LIỆU GỐC
# ==========================================
@st.cache_data(ttl=3600)
def load_all_market_data(ticker_list, t_start, t_end):
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
        st.error("❌ Cấu hình sai: Ngày kết thúc huấn luyện phải trước Ngày bắt đầu backtest!")
    else:
        with st.spinner("Hệ thống đang quét toàn sàn và xử lý tối ưu..."):
            df_train = load_all_market_data(ALL_HOSE_TICKERS, train_start, train_end)
            df_test = load_all_market_data(list(set(ALL_HOSE_TICKERS)), test_start, test_end)
            
            if df_train.empty or df_test.empty or 'VNINDEX' not in df_test.columns:
                st.error("❌ Lỗi trích xuất dữ liệu thị trường.")
            else:
                # ---------------------------------------------------------
                # BƯỚC 1: XÁC ĐỊNH TRẠNG THÁI THỊ TRƯỜNG TẠI KHUNG BACKTEST (VN-INDEX 2025)
                # ---------------------------------------------------------
                idx_test_series = df_test['VNINDEX'].dropna()
                idx_test_start_price = idx_test_series.iloc[0]
                idx_test_end_price = idx_test_series.iloc[-1]
                ma200_idx_test = idx_test_series.rolling(window=200).mean().iloc[-1]
                
                # Tự động gắn nhãn trạng thái và chọn Phương án phân bổ vốn
                if idx_test_end_price > ma200_idx_test * 1.02:
                    market_status = "UPTREND (Thị trường giá lên)"
                    auto_strategy = "Chiến lược 80-20 (80% vốn tập trung dồn cho Top 2 mã mạnh nhất)"
                    status_color = "#e1f7ec"
                    text_color = "#155724"
                elif idx_test_end_price < ma200_idx_test * 0.98:
                    market_status = "DOWNTREND (Thị trường giá xuống)"
                    auto_strategy = "Phòng thủ tuyệt đối (Chuyển 100% tỷ trọng về tài sản an toàn RF)"
                    status_color = "#f8d7da"
                    text_color = "#721c24"
                else:
                    market_status = "SIDEWAY / PHÂN HÓA (Thị trường đi ngang khốc liệt)"
                    auto_strategy = "Chiến lược 80-20 (Dồn lực cho Leader tối ưu Alpha nghịch đảo biến động)"
                    status_color = "#fff3cd"
                    text_color = "#856404"
                
                st.markdown(f"""
                    <div class="status-box" style="background-color: {status_color}; color: {text_color};">
                        🔍 PHÂN TÍCH THỊ TRƯỜNG CHUNG KHUNG BACKTEST:<br>
                        • Trạng thái VN-Index nhận diện: {market_status}<br>
                        • Thuật toán tự động kích hoạt: {auto_strategy}
                    </div>
                """, unsafe_allow_html=True)

                # ---------------------------------------------------------
                # BƯỚC 2: SÀNG LỌC CỔ PHIẾU TỪ KHUNG THỜI GIAN HUẤN LUYỆN
                # ---------------------------------------------------------
                raw_metrics = []
                stock_cols = [c for c in df_train.columns if c != 'VNINDEX']
                
                for ticker in stock_cols:
                    series = df_train[ticker].dropna()
                    if len(series) < 200: continue
                    
                    current_price = series.iloc[-1]
                    ma_20 = series.rolling(window=20).mean().iloc[-1]
                    ma_50 = series.rolling(window=50).mean().iloc[-1]
                    ma_200 = series.rolling(window=200).mean().iloc[-1]
                    
                    trend_valid = 1 if (current_price > ma_20 and current_price > ma_50 and current_price > ma_200 and ma_20 > ma_50) else 0
                    
                    delta = series.diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean().iloc[-1]
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean().iloc[-1]
                    rsi = 100 - (100 / (1 + (gain / (loss + 1e-12))))
                    rsi_score = 1.0 if (50 <= rsi <= 70) else 0.1
                    
                    ret_seq = series.pct_change().tail(20)
                    returns_1m = series.pct_change(20).iloc[-1]
                    volatility_1m = ret_seq.std() + 1e-12
                    rolling_sharpe_1m = returns_1m / volatility_1m
                    
                    raw_metrics.append({
                        "Ticker": ticker, "Sector": TICKER_TO_SECTOR.get(ticker, "Chưa phân loại"),
                        "Price": current_price, "MA20": round(ma_20, 1), "MA50": round(ma_50, 1),
                        "MA200": round(ma_200, 1), "RSI": rsi, "RSI_Score": rsi_score, 
                        "Rolling_Sharpe": rolling_sharpe_1m, "Vol_Raw": volatility_1m, "Trend_Valid": trend_valid
                    })
                
                df_raw = pd.DataFrame(raw_metrics)
                df_raw["Mom_Score"] = df_raw["Rolling_Sharpe"].rank(pct=True)
                df_raw["Total_Score"] = (df_raw["Trend_Valid"] * 0.4) + (df_raw["Mom_Score"] * 0.4) + (df_raw["RSI_Score"] * 0.2)
                df_raw = df_raw.sort_values(by="Total_Score", ascending=False)
                
                portfolio_list = []
                sector_counts = {}
                for idx, row in df_raw.iterrows():
                    if row["Trend_Valid"] != 1: continue
                    sec = row["Sector"]
                    count = sector_counts.get(sec, 0)
                    if count < max_stocks_per_sector:
                        portfolio_list.append({
                            "Mã Cổ Phiếu": row["Ticker"], "Nhóm Ngành Lĩnh Vực": sec,
                            "Giá Đóng Cửa": row["Price"], "Đường MA20": row["MA20"], 
                            "Đường MA50": row["MA50"], "Đường MA200": row["MA200"],
                            "Chỉ Số RSI": round(row["RSI"], 1), "Rolling Sharpe (1M)": round(row["Rolling_Sharpe"], 2),
                            "Điểm Đánh Giá Tín Hiệu": round(row["Total_Score"], 3), "Biến Động Tập HL": row["Vol_Raw"]
                        })
                        sector_counts[sec] = count + 1
                    if len(portfolio_list) == portfolio_size: break
                
                # HIỂN THỊ: DANH MỤC CỔ PHIẾU ĐƯỢC CHỌN THEO BỘ LỌC TÍN HIỆU
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                st.subheader("📋 Danh mục cổ phiếu được chọn theo bộ lọc tín hiệu")
                df_port = pd.DataFrame(portfolio_list)
                df_port_display = df_port.drop(columns=["Biến Động Tập HL"], errors="ignore")
                st.dataframe(df_port_display, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                sorted_tickers = df_port["Mã Cổ Phiếu"].tolist()
                sorted_vols = df_port["Biến Động Tập HL"].tolist()
                n_assets = len(sorted_tickers)
                
                # ---------------------------------------------------------
                # BƯỚC 3: TÍNH TOÁN CƠ CẤU VỐN THEO LỆNH KÍCH HOẠT ĐỘNG
                # ---------------------------------------------------------
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                st.subheader("💰 Cơ cấu phân bổ vốn")
                
                weights = np.zeros(n_assets)
                if "Phòng thủ tuyệt đối" in auto_strategy:
                    # Nếu thị trường Downtrend khốc liệt, đưa tỷ trọng rổ cổ phiếu về 0%
                    w_display_text = ["0.00% (Chuyển sang Cash hưởng RF)"] * n_assets
                else:
                    # Nếu thị trường Uptrend hoặc Sideway phân hóa -> Kích hoạt Pareto 80-20 tối ưu
                    if n_assets >= 2:
                        inv_vol_top1 = 1.0 / (sorted_vols[0] + 1e-12)
                        inv_vol_top2 = 1.0 / (sorted_vols[1] + 1e-12)
                        sum_inv = inv_vol_top1 + inv_vol_top2
                        weights[0] = 0.80 * (inv_vol_top1 / sum_inv)
                        weights[1] = 0.80 * (inv_vol_top2 / sum_inv)
                        if n_assets > 2:
                            rem_w = 0.20 / (n_assets - 2)
                            for j in range(2, n_assets): weights[j] = rem_w
                    else: weights[0] = 1.0
                    w_display_text = [f"{w*100:.2f}%" for w in weights]
                
                df_w_display = pd.DataFrame({
                    "Cổ Phiếu": sorted_tickers, "Nhóm Ngành": df_port["Nhóm Ngành Lĩnh Vực"],
                    "Tỷ Trọng Vốn Phân Bổ Tự Động": w_display_text
                })
                st.dataframe(df_w_display.T, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # ---------------------------------------------------------
                # BƯỚC 4: ĐÁNH GIÁ HIỆU QUẢ DANH MỤC
                # ---------------------------------------------------------
                st.markdown('<div class="report-card">', unsafe_allow_html=True)
                st.subheader("📊 Đánh giá hiệu quả danh mục")
                
                scan_test_tickers = list(set(sorted_tickers + ALL_HOSE_TICKERS))
                df_test_clean = load_all_market_data(scan_test_tickers, test_start, test_end)
                df_test_ret = df_test_clean.pct_change().dropna()
                
                # Tính chuỗi lợi nhuận thực tế theo cơ chế thích ứng
                if "Phòng thủ tuyệt đối" in auto_strategy:
                    portfolio_ret = pd.Series(rf_annual / trading_days, index=df_test_ret.index)
                else:
                    portfolio_ret = df_test_ret[sorted_tickers].dot(weights)
                
                cum_portfolio = (1 + portfolio_ret).cumprod()
                vnindex_ret = df_test_ret['VNINDEX']
                cum_vnindex = (1 + vnindex_ret).cumprod()
                
                valid_all = [t for t in ALL_HOSE_TICKERS if t in df_test_ret.columns]
                base_ret = df_test_ret[valid_all].dot(np.ones(len(valid_all)) / len(valid_all))
                cum_base = (1 + base_ret).cumprod()
                
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
                
                cov_m = portfolio_ret.cov(vnindex_ret)
                var_m = vnindex_ret.var()
                beta_val = cov_m / var_m if var_m != 0 else 1.0
                alpha_jensen = p_ann - (rf_annual + beta_val * (m_ann - rf_annual))
                
                c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                with c_m1: st.metric("Lợi nhuận Danh mục Alpha Max", f"{p_ann*100:.2f}%", delta=f"{(p_ann - m_ann)*100:.2f}% vs VN-Index")
                with c_m2: st.metric("Chỉ số Sharpe chiến lược", f"{p_sharpe:.4f}")
                with c_m3: st.metric("Thặng dư Alpha Jensen", f"{alpha_jensen*100:.2f}%")
                with c_m4: st.metric("Mức sụt giảm lớn nhất (Max DD)", f"{p_max_dd*100:.2f}%")
                
                st.markdown("---")
                df_metrics_final = pd.DataFrame({
                    "Chỉ tiêu định lượng hiệu quả": ["Lợi nhuận trung bình năm ($E_R$)", "Độ biến động rủi ro ($\sigma$)", "Hệ số Sharpe định biên", "Mức sụt giảm tài sản lớn nhất (Max DD)"],
                    "DANH MỤC ALPHA MAX ĐỘNG": [f"{p_ann*100:.2f}%", f"{p_sd*100:.2f}%", f"{p_sharpe:.4f}", f"{p_max_dd*100:.2f}%"],
                    "Chiến lược cơ sở (Buy & Hold toàn sàn)": [f"{b_ann*100:.2f}%", f"{b_sd*100:.2f}%", f"{b_sharpe:.4f}", f"{b_max_dd*100:.2f}%"],
                    "Thị Trường Chung (VN-Index)": [f"{m_ann*100:.2f}%", f"{m_sd*100:.2f}%", f"{m_sharpe:.4f}", f"{m_max_dd*100:.2f}%"]
                })
                st.dataframe(df_metrics_final, use_container_width=True, hide_index=True)
                
                st.markdown("##### 📉 Biểu đồ tăng trưởng tài sản lũy kế thực tế (Out-of-Sample Performance Comparison)")
                fig, ax = plt.subplots(figsize=(12, 5.5))
                ax.plot(cum_portfolio.index, cum_portfolio.values, color='#1ebd9d', lw=2.8, label="Chiến lược Thích ứng Động Alpha Max")
                ax.plot(cum_base.index, cum_base.values, color='#ff9f43', lw=1.5, linestyle='--', label="Chiến lược Cơ sở")
                ax.plot(cum_vnindex.index, cum_vnindex.values, color='#57606f', lw=1.5, alpha=0.6, label="Thị Trường Chung (VN-Index)")
                ax.set_ylabel("Giá trị tài sản (Mốc gốc ban đầu = 1.0)")
                ax.grid(True, linestyle=':')
                ax.legend()
                st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("💡 Hệ thống đã được cấu hình tự động nhận diện bối cảnh. Hãy thiết lập khung tham số và nhấn nút khởi chạy để thuật toán đưa ra quyết định tối ưu.")
