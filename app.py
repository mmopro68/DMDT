import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from vnstock import *

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Hệ Thống Quét Toàn Thị Trường", layout="wide", page_icon="🌐")
st.title("🌐 Hệ Thống Quét Tín Hiệu Dòng Tiền Toàn Diện Toàn Thị Trường")
st.caption("Thuật toán tự động quét toàn bộ danh sách cổ phiếu niêm yết trên sàn để tìm kiếm cơ hội alpha, loại bỏ định kiến nhóm ngành.")

# ==========================================
# 2. TỰ ĐỘNG LẤY DANH SÁCH MÃ TOÀN THỊ TRƯỜNG (CÓ LỌC ĐIỀU KIỆN)
# ==========================================
@st.cache_data(ttl=86400) # Chỉ tải danh sách công ty 1 lần mỗi ngày
def get_all_market_tickers():
    try:
        # Lấy bảng danh sách toàn bộ doanh nghiệp niêm yết từ vnstock
        df_companies = listing_companies()
        if df_companies is not None and not df_companies.empty:
            # Lọc ưu tiên các mã trên sàn HOSE để đảm bảo tính thanh khoản pháp lý cao
            # Loại bỏ các công ty có ký tự đặc biệt hoặc độ dài mã khác 3 ký tự (chứng quyền, quỹ...)
            df_filtered = df_companies[
                (df_companies['comGroupCode'] == 'HOSE') & 
                (df_companies['ticker'].str.len() == 3)
            ]
            return df_filtered['ticker'].tolist()
    except Exception as e:
        pass
    # Phương án dự phòng nếu API nghẽn không lấy được danh sách động
    return ['HPG', 'SSI', 'VCB', 'TCB', 'VHM', 'VIC', 'FPT', 'MWG', 'VNM', 'VCI', 'HCM', 'MBB', 'STB', 'DXG', 'DIG', 'PDR']

# Khởi tạo rổ cổ phiếu diện rộng
ALL_TICKERS = get_all_market_tickers()

# ==========================================
# 3. THANH ĐIỀU HƯỚNG CẤU HÌNH (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Cấu Hình Tham Số")
st.sidebar.success(f"🔍 Hệ thống đã nhận diện tự động {len(ALL_TICKERS)} mã cổ phiếu trên sàn HOSE để sẵn sàng đưa vào bộ quét.")

# Giới hạn số lượng mã quét tối đa để bảo vệ máy chủ không bị sập (Khuyên dùng: 40-60 mã top thanh khoản hoặc chọn ngẫu nhiên)
max_scan = st.sidebar.slider("Giới hạn số mã quét tối đa để tối ưu tốc độ", min_value=20, max_value=100, value=50, step=10)
scanned_tickers = ALL_TICKERS[:max_scan]

rf_annual = st.sidebar.number_input("Lãi suất phi rủi ro/năm (RF)", min_value=0.0, max_value=0.2, value=0.045, step=0.005)
trading_days = st.sidebar.number_input("Số ngày giao dịch một năm", min_value=100, max_value=300, value=252)

st.sidebar.subheader("📅 Khoảng thời gian dữ liệu")
start_date = st.sidebar.date_input("Ngày bắt đầu", datetime(2022, 1, 1)) # Khuyên dùng từ 2022 để tối ưu hóa thời gian tải diện rộng
end_date = st.sidebar.date_input("Ngày kết thúc", datetime(2025, 12, 31))

st.sidebar.subheader("🎯 Chiến lược phân bổ vốn")
strategy_option = st.sidebar.radio(
    "Chọn phương thức chia tỷ trọng cho các hạt giống hàng đầu:", 
    ["Phân bổ đều (Equal Weight)", "Chiến lược 80-20 (Ưu tiên mã Tín hiệu số 1 & 2)"]
)

# ==========================================
# 4. HÀM TẢI DỮ LIỆU TỪ VNSTOCK (BẢN 0.2.8.8)
# ==========================================
@st.cache_data(ttl=3600)
def load_stock_data(ticker_list, start, end):
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    
    close_prices = {}
    progress_bar = st.progress(0, text="Hệ thống đang nạp dữ liệu lịch sử diện rộng...")
    
    for i, ticker in enumerate(ticker_list):
        try:
            df = stock_historical_data(symbol=ticker, start_date=start_str, end_date=end_str, resolution='1D', type='stock')
            if df is not None and not df.empty:
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.set_index('time')
                df = df.sort_index()
                close_prices[ticker] = pd.to_numeric(df['close'], errors='coerce')
        except:
            continue
        progress_bar.progress((i + 1) / len(ticker_list), text=f"Đang phân tích kỹ thuật mã: {ticker}")
    
    progress_bar.empty()
    if not close_prices:
        return pd.DataFrame()
    return pd.DataFrame(close_prices).dropna(how='all')

# ==========================================
# 5. LOGIC XỬ LÝ CHÍNH KHI BẤM NÚT CHẠY
# ==========================================
if st.sidebar.button("🚀 Kích Hoạt Quét Toàn Thị Trường", type="primary"):
    with st.spinner("Đang xử lý ma trận chỉ báo toán học cho toàn bộ rổ cổ phiếu..."):
        
        df_prices = load_stock_data(scanned_tickers, start_date, end_date)
        
        if df_prices.empty or df_prices.shape[1] < 3:
            st.error("❌ Không thể khởi tạo ma trận dữ liệu. Vui lòng rút ngắn khoảng thời gian test.")
        else:
            st.subheader("📊 Kết Quả Sàng Lọc Tín Hiệu Khách Quan Toàn Thị Trường")
            
            # --- BƯỚC 1: QUÉT VÀ CHẤM ĐIỂM TÍN HIỆU ---
            signal_scores = {}
            
            for ticker in df_prices.columns:
                series = df_prices[ticker].dropna()
                if len(series) < 50:
                    continue
                
                current_price = series.iloc[-1]
                
                # Chỉ báo 1: Xu hướng (MA20)
                ma_20 = series.rolling(window=20).mean().iloc[-1]
                trend_signal = 1 if current_price > ma_20 else 0
                
                # Chỉ báo 2: Động lượng (RSI 14)
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
                rs = gain / (loss + 1e-12)
                rsi = 100 - (100 / (1 + rs))
                rsi_signal = 1.0 if (45 <= rsi <= 70) else (0.4 if rsi < 45 else 0.1)
                
                # Chỉ báo 3: Động năng (Momentum 1 tháng)
                momentum_1m = (series.iloc[-1] / series.iloc[-20]) - 1 if len(series) >= 20 else 0
                momentum_signal = 1 if momentum_1m > 0 else 0
                
                total_score = (trend_signal * 0.4) + (rsi_signal * 0.4) + (momentum_signal * 0.2)
                
                signal_scores[ticker] = {
                    "Giá Hiện Tại": current_price,
                    "Đường MA(20)": round(ma_20, 2),
                    "Chỉ Số RSI(14)": round(rsi, 1),
                    "Hiệu Suất 1 Tháng": f"{momentum_1m*100:.1f}%",
                    "Điểm Tín Hiệu": round(total_score, 2)
                }
            
            df_signals = pd.DataFrame(signal_scores).T.sort_values(by="Điểm Tín Hiệu", ascending=False)
            
            st.markdown("#### 🎯 Top 10 Cổ Phiếu Có Tín Hiệu Kỹ Thuật Đột Phá Nhất")
            st.dataframe(df_signals.head(10), use_container_width=True)
            
            # Trích xuất Top 3 mã đạt điểm tối đa bất kể ngành nghề
            top_signal_tickers = df_signals.index[:3].tolist()
            st.success(f"🏆 Danh mục tối ưu gọi tên Top 3 cổ phiếu xuất sắc nhất thị trường: {', '.join(top_signal_tickers)}")
            
            # --- BƯỚC 2: PHÂN BỔ TỶ TRỌNG VÀ BACKTEST ---
            df_returns = df_prices[top_signal_tickers].pct_change().dropna()
            
            weights = np.zeros(len(top_signal_tickers))
            if strategy_option == "Phân bổ đều (Equal Weight)":
                weights = np.ones(len(top_signal_tickers)) / len(top_signal_tickers)
            else:
                if len(top_signal_tickers) >= 2:
                    weights[0] = 0.8
                    weights[1] = 0.2
                else:
                    weights[0] = 1.0
            
            df_weights = pd.DataFrame({"Mã Cổ Phiếu": top_signal_tickers, "Tỷ Trọng Đầu Tư": weights})
            df_weights["Tỷ Trọng Đầu Tư"] = df_weights["Tỷ Trọng Đầu Tư"].map(lambda x: f"{x*100:.1f}%")
            
            portfolio_returns = df_returns.dot(weights)
            cum_returns = (1 + portfolio_returns).cumprod()
            
            p_mean = portfolio_returns.mean() * trading_days
            p_std = portfolio_returns.std() * np.sqrt(trading_days)
            p_sharpe = (p_mean - rf_annual) / p_std if p_std != 0 else 0
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 📐 Cơ Cấu Phân Bổ Vốn Mới")
                st.dataframe(df_weights, use_container_width=True)
                
                st.markdown("#### 📈 Chỉ Số Hiệu Suất Đo Lường")
                st.metric("Lợi nhuận trung bình năm", f"{p_mean*100:.2f}%")
                st.metric("Độ rủi ro (Độ lệch chuẩn năm)", f"{p_std*100:.2f}%")
                st.metric("Hệ số Sharpe danh mục", f"{p_sharpe:.4f}")
                
            with col2:
                st.markdown("#### 📉 Biểu Đồ Đường Tăng Trưởng Tài Sản Khách Quan")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(cum_returns.index, cum_returns.values, color='purple', lw=2.5, label="Danh Mục Tối Ưu Diện Rộng (Alpha Search)")
                ax.set_xlabel("Thời Gian")
                ax.set_ylabel("Giá Trị Tài Sản (Mốc Gốc = 1.0)")
                ax.grid(True, linestyle='--')
                ax.legend()
                st.pyplot(fig)
else:
    st.info("💡 Điểm mới: Thuật toán đã được tích hợp hàm kết nối động tự động cập nhật danh sách doanh nghiệp niêm yết. Vui lòng nhấn nút để chạy bộ lọc.")
