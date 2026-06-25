import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from vnstock import *

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Tối Ưu Danh Mục Đa Ngành", layout="wide", page_icon="🏛️")
st.title("🏛️ Hệ Thống Lọc Cổ Phiếu Đạt Đỉnh Tín Hiệu Theo Từng Nhóm Ngành")
st.caption("Chiến lược lọc hạt giống hàng đầu của từng lĩnh vực nhằm tối ưu hóa đa dạng hóa cấu trúc vốn và phòng ngừa rủi ro hệ thống.")

# ==========================================
# 2. ĐỊNH NGHĨA BẢN ĐỒ PHÂN NHÓM NGÀNH TOÀN DIỆN
# ==========================================
SECTOR_MAP = {
    "Ngân hàng": ['VCB', 'BID', 'CTG', 'TCB', 'MBB', 'VPB', 'ACB', 'STB', 'HDB', 'SHB'],
    "Bất động sản & Trụ lớn": ['VHM', 'VIC', 'VRE', 'NVL', 'PDR', 'KDH', 'NLG', 'DXG', 'DIG', 'CEO'],
    "Thép & Vật liệu": ['HPG', 'HSG', 'NKG', 'VGS', 'POM'],
    "Chứng khoán": ['SSI', 'VND', 'VCI', 'HCM', 'FTS', 'BSI', 'MBS', 'SHS'],
    "Bán lẻ - Công nghệ - Tiêu dùng": ['FPT', 'MWG', 'MSN', 'VNM', 'FRT', 'DGW']
}

# Gom toàn bộ mã để tải một lần
ALL_TICKERS = []
for tickers in SECTOR_MAP.values():
    ALL_TICKERS.extend(tickers)
ALL_TICKERS = list(set(ALL_TICKERS)) # Loại bỏ mã trùng nếu có

# ==========================================
# 3. THANH ĐIỀU HƯỚNG CẤU HÌNH (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Tham Số Khởi Tạo")
st.sidebar.success(f"📊 Hệ thống đang quản lý cấu trúc phân lớp gồm {len(SECTOR_MAP)} nhóm ngành chiến lược với tổng số {len(ALL_TICKERS)} mã cổ phiếu tiêu biểu.")

rf_annual = st.sidebar.number_input("Lãi suất phi rủi ro/năm (RF)", min_value=0.0, max_value=0.2, value=0.045, step=0.005)
trading_days = st.sidebar.number_input("Số ngày giao dịch một năm", min_value=100, max_value=300, value=252)

st.sidebar.subheader("📅 Thời gian huấn luyện & Backtest")
start_date = st.sidebar.date_input("Ngày bắt đầu", datetime(2022, 1, 1))
end_date = st.sidebar.date_input("Ngày kết thúc", datetime(2025, 12, 31))

st.sidebar.subheader("🎯 Cơ chế phân bổ tỷ trọng")
strategy_option = st.sidebar.radio(
    "Chọn phương thức chia vốn cho các mã đại diện ngành:", 
    ["Phân bổ đều (Equal Weight)", "Phân bổ theo trọng số điểm tín hiệu (Score Weight)"]
)

# ==========================================
# 4. HÀM TẢI DỮ LIỆU TỪ VNSTOCK (ĐÃ ĐỒNG BỘ INDEX CHỐNG LỖI)
# ==========================================
@st.cache_data(ttl=3600)
def load_stock_data(ticker_list, start, end):
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    
    close_prices = {}
    progress_bar = st.progress(0, text="Hệ thống đang đồng bộ ma trận giá đóng cửa đa ngành...")
    
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
if st.sidebar.button("🚀 Kích Hoạt Bộ Lọc Phân Lớp Đa Ngành", type="primary"):
    with st.spinner("Đang xử lý thuật toán tối ưu hóa phân lớp..."):
        
        # Tải dữ liệu toàn bộ rổ mã đa ngành
        df_prices = load_stock_data(ALL_TICKERS, start_date, end_date)
        
        if df_prices.empty or df_prices.shape[1] < 5:
            st.error("❌ Không đủ ma trận dữ liệu lịch sử để thực hiện kiểm thử đa ngành.")
        else:
            # --- BƯỚC 1: CHẤM ĐIỂM TÍN HIỆU TOÀN BỘ CỔ PHIẾU ---
            all_signal_data = {}
            for ticker in df_prices.columns:
                series = df_prices[ticker].dropna()
                if len(series) < 50:
                    continue
                
                current_price = series.iloc[-1]
                
                # 1. Tín hiệu Xu hướng (MA20)
                ma_20 = series.rolling(window=20).mean().iloc[-1]
                trend_signal = 1 if current_price > ma_20 else 0
                
                # 2. Tín hiệu Động lượng (RSI 14)
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
                rs = gain / (loss + 1e-12)
                rsi = 100 - (100 / (1 + rs))
                rsi_signal = 1.0 if (45 <= rsi <= 70) else (0.4 if rsi < 45 else 0.1)
                
                # 3. Tín hiệu Động năng (Momentum 1 tháng)
                momentum_1m = (series.iloc[-1] / series.iloc[-20]) - 1 if len(series) >= 20 else 0
                momentum_signal = 1 if momentum_1m > 0 else 0
                
                # Tính tổng điểm
                total_score = (trend_signal * 0.4) + (rsi_signal * 0.4) + (momentum_signal * 0.2)
                
                all_signal_data[ticker] = {
                    "Giá Hiện Tại": current_price,
                    "RSI(14)": round(rsi, 1),
                    "Hiệu Suất 1 Tháng": f"{momentum_1m*100:.1f}%",
                    "Điểm Tín Hiệu": round(total_score, 2)
                }
            
            df_all_signals = pd.DataFrame(all_signal_data).T
            
            # --- BƯỚC 2: PHÂN THEO TỪNG NGÀNH VÀ CHỌN RA MÃ XUẤT SẮC NHẤT (TOP 1 ĐẠI DIỆN NGÀNH) ---
            st.subheader("🎯 Kết Quả Sàng Lọc Hạt Giống Dẫn Đầu Theo Từng Nhóm Ngành")
            
            selected_portfolio_tickers = []
            selected_scores = []
            portfolio_details = []
            
            # Duyệt qua từng ngành trong bản đồ phân lớp
            for sector_name, sector_tickers in SECTOR_MAP.items():
                # Lấy dữ liệu điểm của các mã thuộc ngành hiện tại có trong ma trận giá
                valid_tickers = [t for t in sector_tickers if t in df_all_signals.index]
                
                if valid_tickers:
                    df_sector = df_all_signals.loc[valid_tickers].sort_values(by="Điểm Tín Hiệu", ascending=False)
                    
                    # Trích xuất mã đứng đầu ngành (Top 1)
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
            st.dataframe(df_portfolio_report, use_container_width=True)
            
            # --- BƯỚC 3: PHÂN BỔ VỐN DANH MỤC ĐA NGÀNH (5 MÃ ĐẠI DIỆN) ---
            st.subheader("📐 Phân Bổ Tỷ Trọng Vốn & Đánh Giá Chỉ Tiêu Quản Trị Hiệu Quả")
            
            # Tính toán tỷ trọng
            num_assets = len(selected_portfolio_tickers)
            weights = np.zeros(num_assets)
            
            if strategy_option == "Phân bổ đều (Equal Weight)":
                weights = np.ones(num_assets) / num_assets
            else:
                # Phân bổ theo tỷ trọng điểm tín hiệu
                total_portfolio_score = sum(selected_scores)
                if total_portfolio_score > 0:
                    weights = np.array(selected_scores) / total_portfolio_score
                else:
                    weights = np.ones(num_assets) / num_assets
            
            df_weights = pd.DataFrame({
                "Mã Đại Diện": selected_portfolio_tickers,
                "Ngành": df_portfolio_report["Nhóm Lĩnh Vực"].tolist(),
                "Tỷ Trọng Phân Bổ Vốn": weights
            })
            
            # --- BƯỚC 4: TÍNH TOÁN CÁC CHỈ TIÊU HIỆU QUẢ (RETURNS, VOLATILITY, SHARPE) ---
            df_returns = df_prices[selected_portfolio_tickers].pct_change().dropna()
            portfolio_returns = df_returns.dot(weights)
            cum_returns = (1 + portfolio_returns).cumprod()
            
            # Annualized Return (Lợi nhuận năm)
            p_mean = portfolio_returns.mean() * trading_days
            # Annualized Volatility (Độ rủi ro danh mục có tính đến ma trận hiệp biến dòng tiền đa ngành)
            p_std = portfolio_returns.std() * np.sqrt(trading_days)
            # Sharpe Ratio
            p_sharpe = (p_mean - rf_annual) / p_std if p_std != 0 else 0
            
            # Hiển thị giao diện kết quả phối hợp
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("##### 📊 Bảng Tỷ Trọng Vốn Danh Mục")
                df_display_weights = df_weights.copy()
                df_display_weights["Tỷ Trọng Phân Bổ Vốn"] = df_display_weights["Tỷ Trọng Phân Bổ Vốn"].map(lambda x: f"{x*100:.2f}%")
                st.dataframe(df_display_weights, use_container_width=True)
                
                st.markdown("##### 📊 Chỉ Số Hiệu Hiệu Quả Tối Ưu")
                st.metric("Lợi nhuận trung bình năm ($E_R$)", f"{p_mean*100:.2f}%")
                st.metric("Độ rủi ro danh mục ($\sigma_P$)", f"{p_std*100:.2f}%")
                st.style = st.metric("Hệ số Sharpe tối ưu ($Sharpe Ratio$)", f"{p_sharpe:.4f}")
                
            with col2:
                st.markdown("##### 📉 Biểu Đồ Đường Tăng Trưởng Tài Sản Tích Lũy Đa Ngành")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(cum_returns.index, cum_returns.values, color='indigo', lw=2.5, label="Danh Mục Đa Ngành Phòng Ngừa Rủi Ro Hệ Thống")
                ax.set_xlabel("Thời Gian Kiểm Thử")
                ax.set_ylabel("Giá Trị Tài Sản Tích Lũy (Mốc Gốc = 1.0)")
                ax.grid(True, linestyle='--')
                ax.legend()
                st.pyplot(fig)
else:
    st.info("💡 Điểm cải tiến lý thuyết: Hệ thống đã thiết lập cấu trúc ma trận phân tầng theo nhóm ngành cốt lõi. Hãy nhấn nút để hệ thống tự động nhặt hạt giống đại diện cấu thành danh mục đa ngành.")
