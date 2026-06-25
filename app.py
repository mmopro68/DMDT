import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from vnstock import *

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN STREAMLIT
# ==========================================
st.set_page_config(page_title="Quản Trị Danh Mục Đầu Tư", layout="wide", page_icon="📈")
st.title("📈 Hệ Thống Quét Tín Hiệu & Tối Ưu Hóa Danh Mục Đầu Tư")
st.caption("Ứng dụng phân tích dòng tiền và tối ưu tỷ trọng cổ phiếu theo các chỉ báo kỹ thuật (MA, RSI, Momentum)")

# ==========================================
# 2. DỮ LIỆU NỀN: MÃ CỔ PHIẾU THEO NGÀNH (Tích hợp từ industry_tickers)
# ==========================================
INDUSTRY_TICKERS = {
    "Thép": ['HPG', 'HSG', 'NKG', 'VGS', 'TVN', 'SMC', 'TLH'],
    "Ngân hàng": ['VCB', 'BID', 'CTG', 'TCB', 'MBB', 'VPB', 'ACB', 'HDB', 'STB', 'SHB'],
    "Bất động sản": ['VHM', 'VIC', 'NVL', 'PDR', 'DXG', 'DIG', 'KDH', 'NLG'],
    "Chứng khoán": ['SSI', 'VND', 'VCI', 'HCM', 'MBS', 'FTS', 'BSI', 'VIX']
}

# ==========================================
# 3. THANH ĐIỀU HƯỚNG CẤU HÌNH (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Cấu Hình Tham Số")
selected_industry = st.sidebar.selectbox("Chọn nhóm ngành phân tích", list(INDUSTRY_TICKERS.keys()))
tickers = INDUSTRY_TICKERS[selected_industry]

rf_annual = st.sidebar.number_input("Lại suất phi rủi ro/năm (RF)", min_value=0.0, max_value=0.2, value=0.045, step=0.005)
trading_days = st.sidebar.number_input("Số ngày giao dịch một năm", min_value=100, max_value=300, value=252)

st.sidebar.subheader("📅 Khoảng thời gian dữ liệu")
start_date = st.sidebar.date_input("Ngày bắt đầu", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("Ngày kết thúc", datetime(2025, 12, 31))

st.sidebar.subheader("🎯 Chiến lược phân bổ vốn")
strategy_option = st.sidebar.radio(
    "Chọn phương thức chia tỷ trọng:", 
    ["Phân bổ đều (Equal Weight)", "Chiến lược 80-20 (Ưu tiên mã Tín hiệu số 1 & 2)"]
)

# ==========================================
# 4. HÀM TẢI DỮ LIỆU TỪ VNSTOCK (BẢN ỔN ĐỊNH 0.2.8.8)
# ==========================================
@st.cache_data(ttl=3600)
def load_stock_data(ticker_list, start, end):
    start_str = start.strftime('%Y-%m-%d')
    end_str = end.strftime('%Y-%m-%d')
    
    close_prices = {}
    progress_bar = st.progress(0, text="Đang kết nối dữ liệu thị trường...")
    
    for i, ticker in enumerate(ticker_list):
        try:
            # Gọi hàm trực tiếp từ vnstock 0.2.8.8
            df = stock_historical_data(symbol=ticker, start_date=start_str, end_date=end_str, resolution='1D', type='stock')
            
            if df is not None and not df.empty:
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df = df.set_index('time')
                    
                df = df.sort_index()
                close_prices[ticker] = pd.to_numeric(df['close'], errors='coerce')
        except Exception as e:
            continue
            
        progress_bar.progress((i + 1) / len(ticker_list), text=f"Đang tải mã: {ticker}")
    
    progress_bar.empty()
    
    if not close_prices:
        return pd.DataFrame()
        
    return pd.DataFrame(close_prices).dropna(how='all')
# ==========================================
# 5. LOGIC XỬ LÝ CHÍNH KHI BẤM NÚT CHẠY
# ==========================================
if st.sidebar.button("🚀 Chạy Phân Tích & Tối Ưu", type="primary"):
    with st.spinner("Hệ thống đang thực hiện tính toán chỉ báo kỹ thuật..."):
        
        # Tải dữ liệu giá đóng cửa lịch sử từ thư viện tài chính
        df_prices = load_stock_data(tickers, start_date, end_date)
        
        if df_prices.empty or df_prices.shape[1] < 2:
            st.error("❌ Không lấy được dữ liệu của nhóm ngành hoặc khoảng thời gian được chọn. Vui lòng kiểm tra lại.")
        else:
            st.subheader(f"📊 Kết quả Quét Tín Hiệu & Lọc Cổ Phiếu: Ngành {selected_industry}")
            
            # --- BƯỚC 1: QUÉT VÀ CHẤM ĐIỂM TÍN HIỆU KỸ THUẬT ---
            signal_scores = {}
            
            for ticker in df_prices.columns:
                series = df_prices[ticker].dropna()
                if len(series) < 50:  # Yêu cầu tối thiểu 50 phiên để tính các chỉ báo kỹ thuật ổn định
                    continue
                
                current_price = series.iloc[-1]
                
                # Chỉ báo 1: Xu hướng (Đường trung bình động MA20)
                ma_20 = series.rolling(window=20).mean().iloc[-1]
                trend_signal = 1 if current_price > ma_20 else 0
                
                # Chỉ báo 2: Động lượng (Chỉ số sức mạnh tương đối RSI 14)
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
                rs = gain / (loss + 1e-12)
                rsi = 100 - (100 / (1 + rs))
                
                # Chấm điểm vùng RSI (Khỏe nhất khi RSI đang tích lũy từ 45 đến 70, tránh vùng quá mua >70)
                rsi_signal = 1.0 if (45 <= rsi <= 70) else (0.4 if rsi < 45 else 0.1)
                
                # Chỉ báo 3: Động năng dòng tiền (Momentum hiệu suất 1 tháng gần nhất - 20 phiên)
                momentum_1m = (series.iloc[-1] / series.iloc[-20]) - 1 if len(series) >= 20 else 0
                momentum_signal = 1 if momentum_1m > 0 else 0
                
                # Tổng hợp Điểm Tín Hiệu (Trọng số cấu hình: 40% Xu hướng, 40% Động lượng, 20% Động năng)
                total_score = (trend_signal * 0.4) + (rsi_signal * 0.4) + (momentum_signal * 0.2)
                
                signal_scores[ticker] = {
                    "Giá Hiện Tại": current_price,
                    "Đường MA(20)": round(ma_20, 2),
                    "Chỉ Số RSI(14)": round(rsi, 1),
                    "Hiệu Suất 1 Tháng": f"{momentum_1m*100:.1f}%",
                    "Điểm Tín Hiệu": round(total_score, 2)
                }
            
            # Chuyển đổi dữ liệu sang dạng bảng DataFrame và sắp xếp giảm dần theo Điểm Tín Hiệu
            df_signals = pd.DataFrame(signal_scores).T.sort_values(by="Điểm Tín Hiệu", ascending=False)
            
            st.markdown("#### 🎯 Bảng xếp hạng trạng thái kỹ thuật của các mã")
            st.dataframe(df_signals, use_container_width=True)
            
            # Lọc ra Top 3 mã đứng đầu danh sách có tín hiệu dòng tiền khỏe nhất
            top_signal_tickers = df_signals.index[:3].tolist()
            st.success(f"🔥 Cổ phiếu lọt bộ lọc có tín hiệu dòng tiền mạnh nhất: {', '.join(top_signal_tickers)}")
            
            # --- BƯỚC 2: PHÂN BỔ TỶ TRỌNG VÀ KIỂM THỬ ĐẦU TƯ (BACKTEST) ---
            df_returns = df_prices[top_signal_tickers].pct_change().dropna()
            
            weights = np.zeros(len(top_signal_tickers))
            if strategy_option == "Phân bổ đều (Equal Weight)":
                weights = np.ones(len(top_signal_tickers)) / len(top_signal_tickers)
            else:
                # Áp dụng quy tắc phân bổ tỷ lệ cố định 80-20 dựa trên thứ hạng tín hiệu
                if len(top_signal_tickers) >= 2:
                    weights[0] = 0.8
                    weights[1] = 0.2
                else:
                    weights[0] = 1.0
            
            # Khởi tạo bảng hiển thị cấu trúc danh mục đầu tư
            df_weights = pd.DataFrame({"Mã Cổ Phiếu": top_signal_tickers, "Tỷ Trọng Đầu Tư": weights})
            df_weights["Tỷ Trọng Đầu Tư"] = df_weights["Tỷ Trọng Đầu Tư"].map(lambda x: f"{x*100:.1f}%")
            
            # Tính toán lợi nhuận tích lũy (Tăng trưởng tài sản) qua chuỗi thời gian
            portfolio_returns = df_returns.dot(weights)
            cum_returns = (1 + portfolio_returns).cumprod()
            
            # Tính toán các chỉ số quản trị hiệu suất danh mục hàng năm (Annualized)
            p_mean = portfolio_returns.mean() * trading_days
            p_std = portfolio_returns.std() * np.sqrt(trading_days)
            p_sharpe = (p_mean - rf_annual) / p_std if p_std != 0 else 0
            
            # Hiển thị kết quả chi tiết lên giao diện ứng dụng (Chia làm 2 cột)
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 📐 Cơ Cấu Phân Bổ Tỷ Trọng")
                st.dataframe(df_weights, use_container_width=True)
                
                st.markdown("#### 📈 Chỉ Số Hiệu Suất Danh Mục")
                st.metric("Lợi nhuận trung bình năm", f"{p_mean*100:.2f}%")
                st.metric("Độ rủi ro (Độ lệch chuẩn năm)", f"{p_std*100:.2f}%")
                st.metric("Hệ số Sharpe danh mục", f"{p_sharpe:.4f}")
                
            with col2:
                st.markdown("#### 📉 Biểu Đồ Tăng Trưởng Tài Sản Tích Lũy")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(cum_returns.index, cum_returns.values, color='dodgerblue', lw=2.5, label="Danh Mục Bộ Lọc Tín Hiệu")
                ax.set_xlabel("Thời Gian")
                ax.set_ylabel("Giá Trị Tài Sản (Mốc Gốc = 1.0)")
                ax.grid(True, linestyle='--')
                ax.legend()
                st.pyplot(fig)
else:
    st.info("💡 Bạn hãy cấu hình các tham số đầu vào ở thanh điều hướng bên trái, sau đó nhấn nút 'Chạy Phân Tích & Tối Ưu' để xem kết quả hoạt động của bộ lọc kỹ thuật.")
