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
st.title("🌐 Hệ Thống Phân Lớp Đa Ngành & Lọc Hạt Giống Toàn Thị Trường (Out-of-Sample Backtest)")
st.caption("Thuật toán tối ưu: Xếp hạng tín hiệu toàn thị trường kết hợp bộ lọc giới hạn tỷ trọng ngành nhằm giảm thiểu rủi ro hệ thống.")

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
                if c in df_comp.columns:
                    col_industry = c
                    break
            
            if col_industry is None:
                raise KeyError("Không tìm thấy cột phân loại ngành.")
                
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
                tickers_in_sector = group[col_ticker].unique().tolist()
                if len(tickers_in_sector) >= 2: 
                    dynamic_map[sector_name] = tickers_in_sector
                    
            return dynamic_map
    except Exception as e:
        pass
    
    return {
        "Ngân hàng": ['VCB', 'BID', 'CTG', 'TCB', 'MBB', 'VPB', 'ACB', 'STB', 'HDB', 'TPB'],
        "Bất động sản": ['VHM', 'VIC', 'VRE', 'NVL', 'PDR', 'KDH', 'NLG', 'DXG', 'DIG'],
        "Thép & Vật liệu": ['HPG', 'HSG', 'NKG'],
        "Dịch vụ tài chính (Chứng khoán)": ['SSI', 'VND', 'VCI', 'HCM', 'FTS', 'BSI'],
        "Bán lẻ - Công nghệ - Tiêu dùng": ['FPT', 'MWG', 'MSN', 'VNM', 'FRT', 'DGW']
    }

DYNAMIC_SECTOR_MAP = get_dynamic_sector_map()
# Tạo bản đồ ngược: Ticker -> Sector Name để tra cứu nhanh xu hướng đa ngành
TICKER_TO_SECTOR = {}
for sector, tickers in DYNAMIC_SECTOR_MAP.items():
    for t in tickers:
        TICKER_TO_SECTOR[t] = sector

# ==========================================
# 3. THANH ĐIỀU HƯỚNG CẤU HÌNH (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Tham Số Khởi Tạo")

all_available_sectors = list(DYNAMIC_SECTOR_MAP.keys())
default_sectors = [s for s in ["Ngân hàng", "Bất động sản", "Thép & Vật liệu", "Dịch vụ tài chính (Chứng khoán)", "Bán lẻ - Công nghệ - Tiêu dùng"] if s in all_available_sectors]
if not default_sectors:
    default_sectors = all_available_sectors[:5]

selected_sectors = st.sidebar.multiselect(
    "Chọn các nhóm ngành đưa vào bộ lọc tổng:",
    options=all_available_sectors,
    default=default_sectors
)

final_scan_tickers = []
for s in selected_sectors:
    final_scan_tickers.extend(DYNAMIC_SECTOR_MAP[s])
final_scan_tickers = list(set(final_scan_tickers))

st.sidebar.success(f"📊 Tổng quy mô quét: {len(final_scan_tickers)} mã cổ phiếu.")

# NÂNG CẤP CHỨC NĂNG 1: Cho phép tự chọn quy mô danh mục từ 5 đến 10 mã
portfolio_size = st.sidebar.slider("Số lượng cổ phiếu trong danh mục", min_value=5, max_value=10, value=5, step=1)

# NÂNG CẤP CHỨC NĂNG 2: Giới hạn trần tỷ trọng ngành để phòng vệ rủi ro hệ thống
max_stocks_per_sector = st.sidebar.slider("Số mã tối đa được phép trong cùng 1 ngành", min_value=1, max_value=4, value=2, 
                                          help="Ví dụ bằng 2: Ngành Ngân hàng dù khỏe đến mấy cũng chỉ được tối đa 2 mã lọt vào danh mục, nhường chỗ cho ngành khác để tránh rủi ro hệ thống.")

rf_annual = st.sidebar.number_input("Lãi suất phi rủi ro/năm (RF)", min_value=0.0, max_value=0.2, value=0.045, step=0.005)
trading_days = st.sidebar.number_input("Số ngày giao dịch một năm", min_value=100, max_value=300, value=252)

st.sidebar.subheader("📅 Giai đoạn Huấn luyện (In-Sample)")
train_start = st.sidebar.date_input("Ngày bắt đầu huấn luyện", datetime(2022, 1, 1))
train_end = st.sidebar.date_input("Ngày kết thúc huấn luyện", datetime(2024, 12, 31))

st.sidebar.subheader("📅 Giai đoạn Kiểm thử (Out-of-Sample)")
test_start = st.sidebar.date_input("Ngày bắt đầu kiểm thử", datetime(2025, 1, 1))
test_end = st.sidebar.date_input("Ngày kết thúc kiểm thử", datetime(2025, 12, 31))

st.sidebar.subheader("🎯 Cơ chế phân bổ tỷ trọng")
strategy_option = st.sidebar.radio(
    "Chọn phương thức chia vốn cho danh mục đề xuất:", 
    [
        "Phân bổ đều (Equal Weight)", 
        "Phân bổ theo trọng số điểm tín hiệu (Score Weight)",
        "Chiến lược 80-20 (80% vốn tập trung dồn cho Top 2 mã mạnh nhất)"
    ]
)

# ==========================================
# 4. HÀM TẢI DỮ LIỆU TỪ VNSTOCK
# ==========================================
@st.cache_data(ttl=3600)
def load_all_periods_data(ticker_list, t_start, t_end):
    start_str = t_start.strftime('%Y-%m-%d')
    end_str = t_end.strftime('%Y-%m-%d')
    series_dict = {}
    
    for ticker in ticker_list:
        try:
            df = stock_historical_data(symbol=ticker, start_date=start_str, end_date=end_str, resolution='1D', type='stock')
            if df is not None and not df.empty:
                date_col = 'time' if 'time' in df.columns else ('date' if 'date' in df.columns else None)
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    df = df.drop_duplicates(subset=[date_col]).set_index(date_col).sort_index()
                    series_dict[ticker] = pd.to_numeric(df['close'], errors='coerce')
        except:
            continue
            
    try:
        df_vnindex = stock_historical_data(symbol='VNINDEX', start_date=start_str, end_date=end_str, resolution='1D', type='index')
        if df_vnindex is not None and not df_vnindex.empty:
            date_col = 'time' if 'time' in df_vnindex.columns else ('date' if 'date' in df_vnindex.columns else None)
            if date_col:
                df_vnindex[date_col] = pd.to_datetime(df_vnindex[date_col])
                df_vnindex = df_vnindex.drop_duplicates(subset=[date_col]).set_index(date_col).sort_index()
                series_dict['VNINDEX'] = pd.to_numeric(df_vnindex['close'], errors='coerce')
    except:
        pass
        
    if not series_dict:
        return pd.DataFrame()
    df_final = pd.concat(series_dict, axis=1).sort_index().ffill().bfill()
    return df_final

def calculate_max_drawdown(cum_returns_series):
    rolling_max = cum_returns_series.cummax()
    drawdowns = (cum_returns_series - rolling_max) / rolling_max
    return drawdowns.min()

# ==========================================
# 5. LOGIC THỰC THI CHÍNH KHI BẤM NÚT
# ==========================================
if st.sidebar.button("🚀 Kích Hoạt Mô Hình Tối Ưu Tách Biệt", type="primary"):
    if train_end >= test_start:
        st.error("❌ Lỗi cấu hình thời gian: Ngày kết thúc huấn luyện phải trước Ngày bắt đầu kiểm thử!")
    else:
        with st.spinner("Đang tính toán ma trận tín hiệu toàn thị trường..."):
            df_train = load_all_periods_data(final_scan_tickers, train_start, train_end)
            
            if df_train.empty:
                st.error("❌ Không tải được dữ liệu cho giai đoạn huấn luyện.")
            else:
                # --- BƯỚC 1: CHẤM ĐIỂM TÍN HIỆU TOÀN BỘ CỔ PHIẾU KHÔNG PHÂN BIỆT NGÀNH ---
                all_signal_data = {}
                stock_cols = [c for c in df_train.columns if c != 'VNINDEX']
                
                for ticker in stock_cols:
                    series = df_train[ticker].dropna()
                    if len(series) < 50: continue
                    
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
                        "Mã Cổ Phiếu": ticker,
                        "Thuộc Ngành": TICKER_TO_SECTOR.get(ticker, "Khác"),
                        "Giá Chốt HL": current_price,
                        "Xu Hướng MA20": "TĂNG (Up)" if trend_signal == 1 else "GIẢM (Down)",
                        "Chỉ Số RSI Chi Tiết": round(rsi, 1),
                        "Động Lượng 1 Tháng": f"{momentum_1m*100:.1f}%",
                        "Điểm Tín Hiệu Tổng Hợp": round(total_score, 2),
                        "Hợp Lệ Đầu Tư": trend_signal
                    }
                
                df_market_ranking = pd.DataFrame(all_signal_data).T
                # Sắp xếp toàn bộ thị trường theo điểm số tín hiệu kỹ thuật từ cao xuống thấp
                df_market_ranking = df_market_ranking.sort_values(by="Điểm Tín Hiệu Tổng Hợp", ascending=False)
                
                # --- BƯỚC 2: THUẬT TOÁN SÀNG LỌC ĐA NGÀNH LINH HOẠT TRÁNH RỦI RO HỆ THỐNG ---
                portfolio_details = []
                sector_counter = {} # Đếm số lượng mã của từng ngành đã lọt vào danh mục
                
                for idx, row in df_market_ranking.iterrows():
                    # Chỉ chấp nhận những mã có tín hiệu nằm trong kênh xu hướng tăng để bảo toàn vốn
                    if row["Hợp Lệ Đầu Tư"] != 1:
                        continue
                        
                    current_sector = row["Thuộc Ngành"]
                    current_count = sector_counter.get(current_sector, 0)
                    
                    # KIỂM TRA TIÊU CHÍ ĐA DẠNG HÓA: Nếu ngành này chưa vượt quá giới hạn trần quy định
                    if current_count < max_stocks_per_sector:
                        portfolio_details.append({
                            "Thứ Hạng Toàn Sàn": len(portfolio_details) + 1,
                            "Mã Cổ Phiếu": row["Mã Cổ Phiếu"],
                            "Nhóm Ngành Lĩnh Vực": current_sector,
                            "Giá Chốt Tập HL": row["Giá Chốt HL"],
                            "Xu Hướng MA20": row["Xu Hướng MA20"],
                            "Chỉ Số RSI Chi Tiết": row["Chỉ Số RSI Chi Tiết"],
                            "Động Lượng 1 Tháng": row["Động Lượng 1 Tháng"],
                            "Điểm Tín Hiệu Kỹ Thuật": row["Điểm Tín Hiệu Tổng Hợp"]
                        })
                        # Cập nhật số lượng của ngành này trong rổ danh mục
                        sector_counter[current_sector] = current_count + 1
                    
                    # Dừng lại khi đã gom đủ số lượng cổ phiếu mà người dùng cấu hình qua Slider
                    if len(portfolio_details) == portfolio_size:
                        break
                
                # HIỂN THỊ KẾT QUẢ DANH MỤC LỌC ĐƯỢC
                st.subheader(f"🎯 Danh Mục {len(portfolio_details)} Hạt Giống Xuất Sắc Nhất Thỏa Mãn Bộ Lọc Đa Ngành Phòng Vệ")
                
                if len(portfolio_details) < portfolio_size:
                    st.warning(f"⚠️ Thị trường phân hóa mạnh, chỉ tìm được {len(portfolio_details)} mã thỏa mãn điều kiện Up-trend thay vì mục tiêu {portfolio_size} mã.")
                
                if not portfolio_details:
                    st.error("❌ Không tìm thấy mã cổ phiếu nào đủ điều kiện xu hướng tăng.")
                else:
                    df_portfolio_report = pd.DataFrame(portfolio_details)
                    st.dataframe(df_portfolio_report, use_container_width=True, hide_index=True)
                    
                    sorted_tickers = df_portfolio_report["Mã Cổ Phiếu"].tolist()
                    sorted_scores = df_portfolio_report["Điểm Tín Hiệu Kỹ Thuật"].tolist()
                    num_assets = len(sorted_tickers)
                    
                    # --- BƯỚC 3: PHÂN BỔ TỶ TRỌNG VỐN NHẤT QUÁN THEO QUY MÔ DANH MỤC TÙY BIẾN ---
                    weights = np.zeros(num_assets)
                    if strategy_option == "Phân bổ đều (Equal Weight)":
                        weights = np.ones(num_assets) / num_assets
                    elif strategy_option == "Phân bổ theo trọng số điểm tín hiệu (Score Weight)":
                        total_portfolio_score = sum(sorted_scores)
                        weights = np.array(sorted_scores) / total_portfolio_score if total_portfolio_score > 0 else np.ones(num_assets) / num_assets
                    elif strategy_option == "Chiến lược 80-20 (80% vốn tập trung dồn cho Top 2 mã mạnh nhất)":
                        # Cấu trúc bậc thang chuẩn cho danh mục tổng quát N mã (N >= 5)
                        if num_assets >= 2:
                            weights[0] = 0.40  # 40% vốn cho ngôi vương điểm số
                            weights[1] = 0.40  # 40% vốn cho á quân điểm số
                            
                            # 20% còn lại chia đều cho các mã vệ tinh đa dạng hóa ở phía sau
                            remaining_weight = 0.20 / (num_assets - 2) if num_assets > 2 else 0.0
                            for j in range(2, num_assets): 
                                weights[j] = remaining_weight
                        else:
                            weights[0] = 1.0

                    # Hiển thị tỷ trọng phân bổ vốn trực quan dưới dạng bảng nhỏ
                    df_weights_display = pd.DataFrame({
                        "Cổ Phiếu": sorted_tickers,
                        "Ngành": df_portfolio_report["Nhóm Ngành Lĩnh Vực"],
                        "Tỷ Trọng Vốn Phân Bổ": [f"{w*100:.2f}%" for w in weights]
                    })
                    st.markdown("##### 💰 Chi Tiết Cơ Cấu Phân Bổ Vốn Hiện Tại")
                    st.dataframe(df_weights_display.T, use_container_width=True)

                    # --- BƯỚC 4: MANG DANH MỤC ĐI KIỂM THỬ ĐỘC LẬP TRÊN GIAI ĐOẠN OUT-OF-SAMPLE ---
                    with st.spinner("Đang tiến hành kiểm thử độc lập Out-of-Sample trên tập dữ liệu mới..."):
                        df_test = load_all_periods_data(final_scan_tickers, test_start, test_end)
                        
                        if df_test.empty or 'VNINDEX' not in df_test.columns:
                            st.error("❌ Không lấy được dữ liệu thị trường cho giai đoạn kiểm thử Out-of-Sample.")
                        else:
                            df_test_returns = df_test.pct_change().dropna()
                            
                            # Hiệu suất danh mục tối ưu
                            portfolio_test_returns = df_test_returns[sorted_tickers].dot(weights)
                            cum_portfolio = (1 + portfolio_test_returns).cumprod()
                            
                            # Hiệu suất thị trường
                            vnindex_test_returns = df_test_returns['VNINDEX']
                            cum_vnindex = (1 + vnindex_test_returns).cumprod()
                            
                            # Hiệu suất chiến lược cơ sở (Buy & Hold toàn bộ rổ ngành quét lúc ban đầu)
                            valid_test_tickers = [t for t in final_scan_tickers if t in df_test_returns.columns]
                            base_weights = np.ones(len(valid_test_tickers)) / len(valid_test_tickers)
                            base_test_returns = df_test_returns[valid_test_tickers].dot(base_weights)
                            cum_base = (1 + base_test_returns).cumprod()
                            
                            # --- BƯỚC 5: TÍNH CÁC CHỈ TIÊU ĐỊNH LƯỢNG THỰC TẾ ---
                            p_mean = portfolio_test_returns.mean() * trading_days
                            m_mean = vnindex_test_returns.mean() * trading_days
                            b_mean = base_test_returns.mean() * trading_days
                            
                            p_std = portfolio_test_returns.std() * np.sqrt(trading_days)
                            m_std = vnindex_test_returns.std() * np.sqrt(trading_days)
                            b_std = base_test_returns.std() * np.sqrt(trading_days)
                            
                            p_sharpe = (p_mean - rf_annual) / p_std if p_std != 0 else 0
                            m_sharpe = (m_mean - rf_annual) / m_std if m_std != 0 else 0
                            b_sharpe = (b_mean - rf_annual) / b_std if b_std != 0 else 0
                            
                            covariance = portfolio_test_returns.cov(vnindex_test_returns)
                            market_variance = vnindex_test_returns.var()
                            beta = covariance / market_variance if market_variance != 0 else 1.0
                            alpha_jensen = p_mean - (rf_annual + beta * (m_mean - rf_annual))
                            
                            p_max_dd = calculate_max_drawdown(cum_portfolio)
                            m_max_dd = calculate_max_drawdown(cum_vnindex)
                            b_max_dd = calculate_max_drawdown(cum_base)
                            
                            # --- BƯỚC 6: HIỂN THỊ MA TRẬN KẾT QUẢ ĐỊNH LƯỢNG KHOA HỌC ---
                            st.subheader(f"📐 Ma Trận Hiệu Quả Thực Tế Giai Đoạn Out-of-Sample ({test_start.strftime('%m/%Y')} - {test_end.strftime('%m/%Y')})")
                            
                            df_compare_metrics = pd.DataFrame({
                                "Chỉ tiêu hiệu quả (Thực tế)": ["Lợi nhuận TB Năm ($E_R$)", "Độ rủi ro biến động ($\sigma$)", "Hệ số Sharpe Ratio", "Mức sụt giảm cực đại (Max DD)"],
                                "Danh Mục Thuật Toán Mới (Xếp Hạng Toàn Sàn + Đa Ngành)": [f"{p_mean*100:.2f}%", f"{p_std*100:.2f}%", f"{p_sharpe:.4f}", f"{p_max_dd*100:.2f}%"],
                                "Chiến Lược Cơ Sở (Buy & Hold Toàn Rổ)": [f"{b_mean*100:.2f}%", f"{b_std*100:.2f}%", f"{b_sharpe:.4f}", f"{b_max_dd*100:.2f}%"],
                                "Thị Trường Chung (VN-Index)": [f"{m_mean*100:.2f}%", f"{m_std*100:.2f}%", f"{m_sharpe:.4f}", f"{m_max_dd*100:.2f}%"]
                            })
                            
                            col_m1, col_m2 = st.columns([3, 2])
                            with col_m1:
                                st.dataframe(df_compare_metrics, use_container_width=True, hide_index=True)
                            with col_m2:
                                st.markdown("##### 🔍 Chỉ Số Thực Tế Theo Mô Hình CAPM")
                                st.metric("Hệ số Beta thực tế ($\beta$)", f"{beta:.3f}", help="Mức biến động thực tế của danh mục so với VN-Index.")
                                st.metric("Hệ số Alpha Jensen thực tế ($\alpha$)", f"{alpha_jensen*100:.2f}%", help="Lợi nhuận thặng dư thực tế kiếm được.")
                            
                            st.markdown("##### 📉 Biểu Đồ Tăng Trưởng Tài Sản Độc Lập Trên Tập Kiểm Thử (Out-of-Sample Asset Growth)")
                            fig, ax = plt.subplots(figsize=(12, 5.5))
                            ax.plot(cum_portfolio.index, cum_portfolio.values, color='teal', lw=2.5, label="Danh Mục Thuật Toán Mới (Bộ Lọc Đa Ngành)")
                            ax.plot(cum_base.index, cum_base.values, color='orange', lw=1.8, linestyle='--', label="Chiến Lược Cơ Sở (Buy & Hold Toàn Rổ Quét)")
                            ax.plot(cum_vnindex.index, cum_vnindex.values, color='gray', lw=1.5, alpha=0.7, label="Thị Trường Chung (VN-Index)")
                            ax.set_xlabel("Thời Gian Kiểm Thử độc lập")
                            ax.set_ylabel("Giá Trị Tài Sản Tích Lũy (Mốc Gốc = 1.0)")
                            ax.grid(True, linestyle=':')
                            ax.legend()
                            st.pyplot(fig)
else:
    st.info("💡 Hệ thống đã được tái cấu trúc sang thuật toán xếp hạng toàn thị trường và giới hạn trần đa ngành phòng vệ. Hãy thiết lập các thanh trượt và nhấn nút khởi chạy.")
