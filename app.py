import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from vnstock import *

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Tối Ưu Đa Ngành Toàn Diện", layout="wide", page_icon="🌐")
st.title("🌐 Hệ Thống Phân Lớp Đa Ngành & So Sánh Hiệu Quả Với VN-Index")
st.caption("Thuật toán định lượng sàng lọc hạt giống Alpha, đo lường Alpha, Beta, Max Drawdown so với thị trường chung và danh mục cơ sở.")

# ==========================================
# 2. TỰ ĐỘNG TẢI VÀ PHÂN LOẠI NGÀNH ĐỘNG (CHỐNG LỖI ĐỔI TÊN CỘT & CHỐNG BỎ SÓT)
# ==========================================
@st.cache_data(ttl=86400)
def get_dynamic_sector_map():
    try:
        df_comp = listing_companies()
        if df_comp is not None and not df_comp.empty:
            # Xác định tên cột linh hoạt của vnstock theo các phiên bản cập nhật
            col_ticker = 'ticker' if 'ticker' in df_comp.columns else ('symbol' if 'symbol' in df_comp.columns else df_comp.columns[0])
            col_group = 'comGroupCode' if 'comGroupCode' in df_comp.columns else ('exchange' if 'exchange' in df_comp.columns else None)
            
            col_industry = None
            for c in ['industryName', 'industry', 'ngành', 'nganh']:
                if c in df_comp.columns:
                    col_industry = c
                    break
            
            if col_industry is None:
                raise KeyError("Không tìm thấy cột phân loại ngành trong dữ liệu sàn.")
                
            # Lọc điều kiện sàn HOSE, mã cổ phiếu 3 ký tự và ngành hợp lệ
            df_filtered = df_comp[
                (df_comp[col_group] == 'HOSE') & 
                (df_comp[col_ticker].str.len() == 3) & 
                (df_comp[col_industry].notna())
            ].copy()
            
            df_filtered[col_ticker] = df_filtered[col_ticker].str.strip()
            df_filtered[col_industry] = df_filtered[col_industry].str.strip()
            
            dynamic_map = {}
            grouped = df_filtered.groupby(col_industry)
            
            for sector_name, group in grouped:
                # Quét TOÀN BỘ mã trong ngành trên sàn HOSE, không giới hạn số lượng để chống bỏ sót
                tickers_in_sector = group[col_ticker].unique().tolist()
                if len(tickers_in_sector) >= 2: 
                    dynamic_map[sector_name] = tickers_in_sector
                    
            return dynamic_map
    except Exception as e:
        pass
    
    # Phương án dự phòng (Fallback) an toàn tuyệt đối nếu API sàn lỗi cấu trúc
    return {
        "Ngân hàng": ['VCB', 'BID', 'CTG', 'TCB', 'MBB', 'VPB', 'ACB', 'STB', 'HDB', 'TPB', 'SHB', 'LPB', 'EIB'],
        "Bất động sản": ['VHM', 'VIC', 'VRE', 'NVL', 'PDR', 'KDH', 'NLG', 'DXG', 'DIG', 'CEO', 'TCH', 'HDG'],
        "Thép & Vật liệu": ['HPG', 'HSG', 'NKG', 'VGS', 'HT1'],
        "Dịch vụ tài chính (Chứng khoán)": ['SSI', 'VND', 'VCI', 'HCM', 'FTS', 'BSI', 'MBS', 'SHS', 'VIX'],
        "Bán lẻ - Công nghệ - Tiêu dùng": ['FPT', 'MWG', 'MSN', 'VNM', 'FRT', 'DGW', 'PNJ', 'SAB']
    }

DYNAMIC_SECTOR_MAP = get_dynamic_sector_map()
ALL_DYNAMIC_TICKERS = []
for t_list in DYNAMIC_SECTOR_MAP.values():
    ALL_DYNAMIC_TICKERS.extend(t_list)
ALL_DYNAMIC_TICKERS = list(set(ALL_DYNAMIC_TICKERS))

# ==========================================
# 3. THANH ĐIỀU HƯỚNG CẤU HÌNH (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Tham Số Khởi Tạo")

all_available_sectors = list(DYNAMIC_SECTOR_MAP.keys())
# Đặt mặc định lấy 5 nhóm ngành lớn tiêu biểu nhất của sàn để tối ưu hóa hiệu năng
default_sectors = [s for s in ["Ngân hàng", "Bất động sản", "Thép & Vật liệu", "Dịch vụ tài chính (Chứng khoán)", "Bán lẻ - Công nghệ - Tiêu dùng"] if s in all_available_sectors]
if not default_sectors:
    default_sectors = all_available_sectors[:5]

selected_sectors = st.sidebar.multiselect(
    "Chọn các nhóm ngành đưa vào mô hình:",
    options=all_available_sectors,
    default=default_sectors
)

final_scan_tickers = []
for s in selected_sectors:
    final_scan_tickers.extend(DYNAMIC_SECTOR_MAP[s])
final_scan_tickers = list(set(final_scan_tickers))

st.sidebar.success(f"📊 Thuật toán đang bao phủ {len(selected_sectors)} ngành với tổng số {len(final_scan_tickers)} mã cổ phiếu đang niêm yết.")

rf_annual = st.sidebar.number_input("Lãi suất phi rủi ro/năm (RF)", min_value=0.0, max_value=0.2, value=0.045, step=0.005)
trading_days = st.sidebar.number_input("Số ngày giao dịch một năm", min_value=100, max_value=300, value=252)

st.sidebar.subheader("📅 Thời gian huấn luyện & Backtest")
start_date = st.sidebar.date_input("Ngày bắt đầu", datetime(2023, 1, 1))
end_date = st.sidebar.date_input("Ngày kết thúc", datetime(2025, 12, 31))

st.sidebar.subheader("🎯 Cơ chế phân bổ tỷ trọng")
strategy_option = st.sidebar.radio(
    "Chọn phương thức chia vốn cho các mã đại diện ngành:", 
    [
        "Phân bổ đều (Equal Weight)", 
        "Phân bổ theo trọng số điểm tín hiệu (Score Weight)",
        "Chiến lược 80-20 (80% vốn dồn cho Top 2 ngành mạnh nhất)"
    ]
)

# ==========================================
# 4. HÀM TẢI DỮ LIỆU TỪ VNSTOCK (SỬA TRIỆT ĐỂ LỖI REINDEX BẰNG PD.CONCAT)
# ==========================================
@st.cache_data(ttl=3600)
def load_stock_and_market_data(ticker_list, start, end):
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    
    series_dict = {}
    progress_bar = st.progress(0, text="Hệ thống đang nạp đồng bộ ma trận giá đóng cửa toàn sàn...")
    
    total_len = len(ticker_list) + 1
    for i, ticker in enumerate(ticker_list):
        try:
            df = stock_historical_data(symbol=ticker, start_date=start_str, end_date=end_str, resolution='1D', type='stock')
            if df is not None and not df.empty:
                date_col = 'time' if 'time' in df.columns else ('date' if 'date' in df.columns else None)
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.drop_duplicates(subset=[date_col])
                    df = df.set_index(date_col).sort_index()
                    series_dict[ticker] = pd.to_numeric(df['close'], errors='coerce')
        except:
            continue
        progress_bar.progress((i + 1) / total_len, text=f"Đang đồng bộ chuỗi dữ liệu mã: {ticker}")
    
    # Tải dữ liệu chỉ số VNINDEX làm Benchmark thị trường chung
    try:
        df_vnindex = stock_historical_data(symbol='VNINDEX', start_date=start_str, end_date=end_str, resolution='1D', type='index')
        if df_vnindex is not None and not df_vnindex.empty:
            date_col = 'time' if 'time' in df_vnindex.columns else ('date' if 'date' in df_vnindex.columns else None)
            if date_col:
                df_vnindex[date_col] = pd.to_datetime(df_vnindex[date_col])
                df_vnindex = df_vnindex.drop_duplicates(subset=[date_col])
                df_vnindex = df_vnindex.set_index(date_col).sort_index()
                series_dict['VNINDEX'] = pd.to_numeric(df_vnindex['close'], errors='coerce')
    except:
        pass
        
    progress_bar.empty()
    if not series_dict:
        return pd.DataFrame()
        
    # Gộp dữ liệu an toàn theo cột bằng pd.concat (khắc phục hoàn toàn lỗi Reindex lệch ngày)
    df_final = pd.concat(series_dict, axis=1)
    df_final.index = pd.to_datetime(df_final.index)
    
    # Điền bù dữ liệu thiếu do ngày nghỉ giao dịch lệch nhau giữa các mã
    df_final = df_final.sort_index().ffill().bfill()
    return df_final

def calculate_max_drawdown(cum_returns_series):
    rolling_max = cum_returns_series.cummax()
    drawdowns = (cum_returns_series - rolling_max) / rolling_max
    return drawdowns.min()

# ==========================================
# 5. LOGIC THỰC THI CHÍNH KHI BẤM NÚT
# ==========================================
if st.sidebar.button("🚀 Kích Hoạt Bộ Lọc Phân Lớp Toàn Thị Trường", type="primary"):
    with st.spinner("Đang xử lý thuật toán tối ưu hóa phân lớp và kiểm thử danh mục..."):
        
        df_prices = load_stock_and_market_data(final_scan_tickers, start_date, end_date)
        
        if df_prices.empty or 'VNINDEX' not in df_prices.columns:
            st.error("❌ Không lấy được dữ liệu chỉ số VNINDEX hoặc ma trận giá cổ phiếu.")
        else:
            # --- BƯỚC 1: CHẤM ĐIỂM TÍN HIỆU ĐA NHÂN TỐ ---
            all_signal_data = {}
            stock_cols = [c for c in df_prices.columns if c != 'VNINDEX']
            
            for ticker in stock_cols:
                series = df_prices[ticker].dropna()
                if len(series) < 50:
                    continue
                
                current_price = series.iloc[-1]
                ma_20 = series.rolling(window=20).mean().iloc[-1]
                trend_signal = 1 if current_price > ma_20 else 0
                
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
                rs = gain / (loss + 1e-12)
                rsi = 100 - (100 / (1 + rs))
                rsi_signal = 1.0 if (45 <= rsi <= 70) else (0.4 if rsi < 45 else 0.1)
                
                momentum_1m = (series.iloc[-1] / series.iloc[-20]) - 1 if len(series) >= 20 else 0
                momentum_signal = 1 if momentum_1m > 0 else 0
                
                total_score = (trend_signal * 0.4) + (rsi_signal * 0.4) + (momentum_signal * 0.2)
                
                all_signal_data[ticker] = {
                    "Giá Hiện Tại": current_price,
                    "RSI(14)": round(rsi, 1),
                    "Hiệu Suất 1 Tháng": f"{momentum_1m*100:.1f}%",
                    "Điểm Tín Hiệu": round(total_score, 2)
                }
            
            df_all_signals = pd.DataFrame(all_signal_data).T
            
            # --- BƯỚC 2: PHÂN THEO TỪNG NGÀNH VÀ CHỌN RA MÃ ĐỈNH TÍN HIỆU ---
            st.subheader("🎯 Kết Quả Sàng Lọc Hạt Giống Dẫn Đầu Ngành")
            
            selected_portfolio_tickers = []
            selected_scores = []
            portfolio_details = []
            
            for sector_name in selected_sectors:
                if sector_name in DYNAMIC_SECTOR_MAP:
                    sector_tickers = DYNAMIC_SECTOR_MAP[sector_name]
                    valid_tickers = [t for t in sector_tickers if t in df_all_signals.index]
                    
                    if valid_tickers:
                        df_sector = df_all_signals.loc[valid_tickers].sort_values(by="Điểm Tín Hiệu", ascending=False)
                        if not df_sector.empty:
                            top_ticker = df_sector.index[0]
                            top_score = df_sector.iloc[0]["Điểm Tín Hiệu"]
                            
                            selected_portfolio_tickers.append(top_ticker)
                            selected_scores.append(top_score)
                            
                            portfolio_details.append({
                                "Nhóm Lĩnh Vực": sector_name,
                                "Mã Đại Diện Mạnh Nhất": top_ticker,
                                "Giá Hiện Tại": df_sector.iloc[0]["Giá Hiện Tại"],
                                "Chỉ Số RSI(14)": df_sector.iloc[0]["RSI(14)"],
                                "Hiệu Suất Ngắn Hạn": df_sector.iloc[0]["Hiệu Suất 1 Tháng"],
                                "Điểm Tín Hiệu Kỹ Thuật": top_score
                            })
            
            df_portfolio_report = pd.DataFrame(portfolio_details)
            df_portfolio_report = df_portfolio_report.sort_values(by="Điểm Tín Hiệu Kỹ Thuật", ascending=False).reset_index(drop=True)
            st.dataframe(df_portfolio_report, use_container_width=True)
            
            sorted_tickers = df_portfolio_report["Mã Đại Diện Mạnh Nhất"].tolist()
            sorted_scores = df_portfolio_report["Điểm Tín Hiệu Kỹ Thuật"].tolist()
            num_assets = len(sorted_tickers)
            
            # --- BƯỚC 3: TRỒNG CẤU TRÚC PHÂN BỔ VỐN LINH HOẠT ---
            weights = np.zeros(num_assets)
            if strategy_option == "Phân bổ đều (Equal Weight)":
                weights = np.ones(num_assets) / num_assets
            elif strategy_option == "Phân bổ theo trọng số điểm tín hiệu (Score Weight)":
                total_portfolio_score = sum(sorted_scores)
                weights = np.array(sorted_scores) / total_portfolio_score if total_portfolio_score > 0 else np.ones(num_assets) / num_assets
            elif strategy_option == "Chiến lược 80-20 (80% vốn dồn cho Top 2 ngành mạnh nhất)":
                if num_assets >= 2:
                    weights[0], weights[1] = 0.40, 0.40
                    remaining_weight = 0.20 / (num_assets - 2)
                    for j in range(2, num_assets): 
                        weights[j] = remaining_weight
                else:
                    weights[0] = 1.0

            # --- BƯỚC 4: TÍNH TOÁN BACKTEST VÀ THAM CHIẾU BENCHMARK SONG PHƯƠNG ---
            df_all_returns = df_prices.pct_change().dropna()
            
            # 1. Tỷ suất danh mục Chiến lược
            portfolio_returns = df_all_returns[sorted_tickers].dot(weights)
            cum_portfolio = (1 + portfolio_returns).cumprod()
            
            # 2. Tỷ suất Thị trường VNINDEX
            vnindex_returns = df_all_returns['VNINDEX']
            cum_vnindex = (1 + vnindex_returns).cumprod()
            
            # 3. Tỷ suất Chiến lược Cơ sở (Mua và nắm giữ đều tất cả mã trong rổ quét)
            valid_scan_tickers = [t for t in final_scan_tickers if t in df_all_returns.columns]
            base_weights = np.ones(len(valid_scan_tickers)) / len(valid_scan_tickers)
            base_returns = df_all_returns[valid_scan_tickers].dot(base_weights)
            cum_base = (1 + base_returns).cumprod()
            
            # --- BƯỚC 5: TÍNH TOÁN CÁC CHỈ TIÊU ĐỊNH LƯỢNG NÂNG CAO ---
            p_mean, m_mean, b_mean = portfolio_returns.mean() * trading_days, vnindex_returns.mean() * trading_days, base_returns.mean() * trading_days
            p_std, m_std, b_std = portfolio_returns.std() * np.sqrt(trading_days), vnindex_returns.std() * np.sqrt(trading_days), base_returns.std() * np.sqrt(trading_days)
            
            p_sharpe = (p_mean - rf_annual) / p_std if p_std != 0 else 0
            m_sharpe = (m_mean - rf_annual) / m_std if m_std != 0 else 0
            b_sharpe = (b_mean - rf_annual) / b_std if b_std != 0 else 0
            
            # Hệ số Beta & Alpha Jensen (Mô hình CAPM)
            covariance = portfolio_returns.cov(vnindex_returns)
            market_variance = vnindex_returns.var()
            beta = covariance / market_variance if market_variance != 0 else 1.0
            alpha_jensen = p_mean - (rf_annual + beta * (m_mean - rf_annual))
            
            p_max_dd = calculate_max_drawdown(cum_portfolio)
            m_max_dd = calculate_max_drawdown(cum_vnindex)
            b_max_dd = calculate_max_drawdown(cum_base)
            
            # --- BƯỚC 6: HIỂN THỊ GIAO DIỆN BÁO CÁO TOÀN DIỆN ---
            st.subheader("📐 Ma Trận Chỉ Tiêu So Sánh Đối Chiếu Hiệu Quả Danh Mục Tổng Thể")
            
            df_compare_metrics = pd.DataFrame({
                "Chỉ tiêu quản trị (Năm)": ["Lợi nhuận TB Năm ($E_R$)", "Độ rủi ro biến động ($\sigma$)", "Hệ số Sharpe Ratio", "Mức sụt giảm cực đại (Max DD)"],
                "Danh Mục Chiến Lược (Thuật toán)": [f"{p_mean*100:.2f}%", f"{p_std*100:.2f}%", f"{p_sharpe:.4f}", f"{p_max_dd*100:.2f}%"],
                "Chiến Lược Cơ Sở (Buy & Hold Toàn rổ)": [f"{b_mean*100:.2f}%", f"{b_std*100:.2f}%", f"{b_sharpe:.4f}", f"{b_max_dd*100:.2f}%"],
                "Thị Trường Chung (VN-Index)": [f"{m_mean*100:.2f}%", f"{m_std*100:.2f}%", f"{m_sharpe:.4f}", f"{m_max_dd*100:.2f}%"]
            })
            
            col_m1, col_m2 = st.columns([3, 2])
            with col_m1:
                st.dataframe(df_compare_metrics, use_container_width=True, hide_index=True)
            with col_m2:
                st.markdown("##### 🔍 Chỉ Số Định Giá Rủi Ro CAPM Nâng Cao")
                st.metric("Hệ số Beta ($\beta$ so với VN-Index)", f"{beta:.3f}", help="Đo lường mức độ nhạy cảm của danh mục trước biến động thị trường chung.")
                st.metric("Hệ số Alpha Jensen ($\alpha$ thặng dư)", f"{alpha_jensen*100:.2f}%", help="Lợi nhuận thặng dư vượt ngoài kỳ vọng rủi ro hệ thống.")
            
            st.markdown("##### 📉 Biểu Đồ Trực Quan Đường Tăng Trưởng Tài Sản Tích Lũy Song Phương")
            fig, ax = plt.subplots(figsize=(12, 5.5))
            ax.plot(cum_portfolio.index, cum_portfolio.values, color='teal', lw=2.5, label=f"Danh Mục Chiến Lược ({strategy_option})")
            ax.plot(cum_base.index, cum_base.values, color='orange', lw=1.8, linestyle='--', label="Chiến Lược Cơ Sở (Buy & Hold Toàn Rổ Quét)")
            ax.plot(cum_vnindex.index, cum_vnindex.values, color='gray', lw=1.5, alpha=0.7, label="Thị Trường Chung (VN-Index)")
            ax.set_xlabel("Thời Gian")
            ax.set_ylabel("Giá Trị Tài Sản Tích Lũy (Mốc Gốc = 1.0)")
            ax.grid(True, linestyle=':')
            ax.legend()
            st.pyplot(fig)
else:
    st.info("💡 Hệ thống đã sẵn sàng cấu trúc tối ưu. Vui lòng bấm nút để khởi chạy và ghi nhận số liệu so sánh.")
