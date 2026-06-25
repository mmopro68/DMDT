import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from vnstock import *

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Tối Ưu Đa Ngành Toàn Thị Trường", layout="wide", page_icon="🌐")
st.title("🌐 Hệ Thống Phân Lớp Đa Ngành & Tối Ưu Hóa Toàn Thị Trường")
st.caption("Thuật toán tự động tải toàn bộ cổ phiếu trên sàn, phân loại ngành tự động và sàng lọc hạt giống Alpha không có định kiến chủ quan.")

# ==========================================
# 2. TỰ ĐỘNG TẢI VÀ PHÂN LOẠI NGÀNH ĐỘNG TOÀN THỊ TRƯỜNG
# ==========================================
@st.cache_data(ttl=86400) # Lưu bộ nhớ đệm 1 ngày để tối ưu tốc độ tải danh mục sàn
def get_dynamic_sector_map():
    try:
        # Lấy danh sách toàn bộ doanh nghiệp niêm yết từ vnstock
        df_comp = listing_companies()
        if df_comp is not None and not df_comp.empty:
            # Lọc điều kiện: Sàn HOSE, mã cổ phiếu tiêu chuẩn (3 ký tự) và có thông tin ngành
            df_filtered = df_comp[
                (df_comp['comGroupCode'] == 'HOSE') & 
                (df_comp['ticker'].str.len() == 3) & 
                (df_comp['industryName'].notna())
            ]
            
            # Group các mã theo tên ngành Tiếng Việt
            dynamic_map = {}
            grouped = df_filtered.groupby('industryName')
            
            for sector_name, group in grouped:
                # Mỗi ngành chỉ lấy tối đa 12 mã để tối ưu băng thông tải dữ liệu, tránh sập máy chủ
                tickers_in_sector = group['ticker'].head(12).tolist()
                if len(tickers_in_sector) >= 2: # Chỉ lấy các ngành có từ 2 mã trở lên
                    dynamic_map[sector_name] = tickers_in_sector
                    
            return dynamic_map
    except Exception as e:
        pass
    
    # Phương án dự phòng (Fallback) nếu API vnstock phân loại ngành bị nghẽn
    return {
        "Ngân hàng": ['VCB', 'BID', 'CTG', 'TCB', 'MBB', 'VPB', 'ACB'],
        "Bất động sản": ['VHM', 'VIC', 'VRE', 'NVL', 'PDR', 'KDH', 'NLG'],
        "Thép & Vật liệu": ['HPG', 'HSG', 'NKG'],
        "Dịch vụ tài chính": ['SSI', 'VND', 'VCI', 'HCM', 'FTS'],
        "Bán lẻ & Công nghệ": ['FPT', 'MWG', 'MSN', 'VNM']
    }

# Gọi hàm khởi tạo bản đồ ngành động từ toàn bộ sàn
DYNAMIC_SECTOR_MAP = get_dynamic_sector_map()

# Tập hợp toàn bộ mã động để chuẩn bị tải dữ liệu giá
ALL_DYNAMIC_TICKERS = []
for t_list in DYNAMIC_SECTOR_MAP.values():
    ALL_DYNAMIC_TICKERS.extend(t_list)
ALL_DYNAMIC_TICKERS = list(set(ALL_DYNAMIC_TICKERS))

# ==========================================
# 3. THANH ĐIỀU HƯỚNG CẤU HÌNH (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Tham Số Khởi Tạo")
st.sidebar.success(f"📊 Thuật toán đã tự động phân loại được {len(DYNAMIC_SECTOR_MAP)} nhóm ngành trên sàn HOSE với tổng số {len(ALL_DYNAMIC_TICKERS)} mã cổ phiếu tiêu chuẩn.")

# Giới hạn số ngành quét tối đa để bảo vệ hiệu năng hệ thống Cloud
max_sectors_to_scan = st.sidebar.slider("Giới hạn số nhóm ngành quét tối đa", min_value=3, max_value=len(DYNAMIC_SECTOR_MAP), value=min(6, len(DYNAMIC_SECTOR_MAP)))

# Lọc lại bản đồ ngành theo giới hạn người dùng chọn
selected_sectors = list(DYNAMIC_SECTOR_MAP.keys())[:max_sectors_to_scan]
final_scan_tickers = []
for s in selected_sectors:
    final_scan_tickers.extend(DYNAMIC_SECTOR_MAP[s])
final_scan_tickers = list(set(final_scan_tickers))

rf_annual = st.sidebar.number_input("Lãi suất phi rủi ro/năm (RF)", min_value=0.0, max_value=0.2, value=0.045, step=0.005)
trading_days = st.sidebar.number_input("Số ngày giao dịch một năm", min_value=100, max_value=300, value=252)

st.sidebar.subheader("📅 Thời gian huấn luyện & Backtest")
start_date = st.sidebar.date_input("Ngày bắt đầu", datetime(2023, 1, 1)) # Khuyên dùng từ 2023 để tối ưu hóa bộ nhớ quét diện rộng
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
# 4. HÀM TẢI DỮ LIỆU TỪ VNSTOCK (ĐÃ CHUẨN HÓA ĐỒNG BỘ INDEX CHỐNG LỖI)
# ==========================================
@st.cache_data(ttl=3600)
def load_stock_data(ticker_list, start, end):
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    
    close_prices = {}
    progress_bar = st.progress(0, text="Hệ thống đang đồng bộ ma trận giá đóng cửa toàn sàn...")
    
    for i, ticker in enumerate(ticker_list):
        try:
            df = stock_historical_data(symbol=ticker, start_date=start_str, end_date=end_str, resolution='1D', type='stock')
            if df is not None and not df.empty:
                date_col = 'time' if 'time' in df.columns else ('date' if 'date' in df.columns else None)
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d')
                    df = df.drop_duplicates(subset=[date_col])
                    df = df.set_index(date_col)
                    df = df.sort_index()
                    close_prices[ticker] = pd.to_numeric(df['close'], errors='coerce')
        except:
            continue
        progress_bar.progress((i + 1) / len(ticker_list), text=f"Đang phân tích chuỗi dữ liệu mã: {ticker}")
    
    progress_bar.empty()
    if not close_prices:
        return pd.DataFrame()
        
    df_final = pd.DataFrame(close_prices)
    df_final.index = pd.to_datetime(df_final.index)
    return df_final.sort_index().dropna(how='all')

# ==========================================
# 5. LOGIC THỰC THI CHÍNH KHI BẤM NÚT
# ==========================================
if st.sidebar.button("🚀 Kích Hoạt Bộ Lọc Phân Lớp Toàn Thị Trường", type="primary"):
    with st.spinner("Đang xử lý ma trận chỉ báo toán học cho toàn bộ sàn HOSE..."):
        
        # Tải dữ liệu toàn bộ các mã thuộc các ngành đã chọn lọc động
        df_prices = load_stock_data(final_scan_tickers, start_date, end_date)
        
        if df_prices.empty or df_prices.shape[1] < 3:
            st.error("❌ Không đủ ma trận dữ liệu lịch sử để thực hiện kiểm thử diện rộng.")
        else:
            # --- BƯỚC 1: CHẤM ĐIỂM TÍN HIỆU TOÀN BỘ CỔ PHIẾU ---
            all_signal_data = {}
            for ticker in df_prices.columns:
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
            
            # --- BƯỚC 2: PHÂN THEO TỪNG NGÀNH ĐỘNG VÀ CHỌN RA MÃ XUẤT SẮC NHẤT ---
            st.subheader("🎯 Kết Quả Sàng Lọc Hạt Giống Dẫn Đầu Ngành Được Cập Nhật Tự Động")
            
            selected_portfolio_tickers = []
            selected_scores = []
            portfolio_details = []
            
            for sector_name in selected_sectors:
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
                            "Nhóm Lĩnh Vực Hệ Thống Xếp": sector_name,
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
            
            # --- BƯỚC 3: PHÂN BỔ TỶ TRỌNG VỐN LINH HOẠT ---
            st.subheader("📐 Phân Bổ Tỷ Trọng Vốn & Đánh Giá Chỉ Tiêu Quản Trị Hiệu Quả Đa Ngành Toàn Diện")
            
            weights = np.zeros(num_assets)
            
            if strategy_option == "Phân bổ đều (Equal Weight)":
                weights = np.ones(num_assets) / num_assets
                
            elif strategy_option == "Phân bổ theo trọng số điểm tín hiệu (Score Weight)":
                total_portfolio_score = sum(sorted_scores)
                if total_portfolio_score > 0:
                    weights = np.array(sorted_scores) / total_portfolio_score
                else:
                    weights = np.ones(num_assets) / num_assets
                    
            elif strategy_option == "Chiến lược 80-20 (80% vốn dồn cho Top 2 ngành mạnh nhất)":
                if num_assets >= 2:
                    weights[0] = 0.40
                    weights[1] = 0.40
                    remaining_weight = 0.20 / (num_assets - 2)
                    for j in range(2, num_assets):
                        weights[j] = remaining_weight
                else:
                    weights[0] = 1.0

            df_weights = pd.DataFrame({
                "Mã Đại Diện": sorted_tickers,
                "Ngành Phân Phối": df_portfolio_report["Nhóm Lĩnh Vực Hệ Thống Xếp"].tolist(),
                "Tỷ Trọng Phân Bổ Vốn": weights
            })
            
            # --- BƯỚC 4: TÍNH TOÁN CÁC CHỈ TIÊU HIỆU QUẢ ---
            df_returns = df_prices[sorted_tickers].pct_change().dropna()
            portfolio_returns = df_returns.dot(weights)
            cum_returns = (1 + portfolio_returns).cumprod()
            
            p_mean = portfolio_returns.mean() * trading_days
            p_std = portfolio_returns.std() * np.sqrt(trading_days)
            p_sharpe = (p_mean - rf_annual) / p_std if p_std != 0 else 0
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("##### 📊 Bảng Tỷ Trọng Vốn Danh Mục Động")
                df_display_weights = df_weights.copy()
                df_display_weights["Tỷ Trọng Phân Bổ Vốn"] = df_display_weights["Tỷ Trọng Phân Bổ Vốn"].map(lambda x: f"{x*100:.2f}%")
                st.dataframe(df_display_weights, use_container_width=True)
                
                st.markdown("##### 📊 Chỉ Số Hiệu Hiệu Quả Tối Ưu")
                st.metric("Lợi nhuận trung bình năm ($E_R$)", f"{p_mean*100:.2f}%")
                st.metric("Độ rủi ro danh mục ($\sigma_P$)", f"{p_std*100:.2f}%")
                st.metric("Hệ số Sharpe tối ưu ($Sharpe Ratio$)", f"{p_sharpe:.4f}")
                
            with col2:
                st.markdown("##### 📉 Biểu Đồ Đường Tăng Trưởng Tài Sản Đa Ngành Tự Động")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(cum_returns.index, cum_returns.values, color='teal', lw=2.5, label="Danh Mục Đa Ngành Động 100% Khách Quan")
                ax.set_xlabel("Thời Gian Kiểm Thử")
                ax.set_ylabel("Giá Trị Tài Sản Tích Lũy (Mốc Gốc = 1.0)")
                ax.grid(True, linestyle='--')
                ax.legend()
                st.pyplot(fig)
else:
    st.info("💡 Mô hình đã được nâng cấp lên cơ chế Tự động hóa phân lớp ngành toàn thị trường. Hãy nhấn nút để khởi chạy bộ lọc động.")
