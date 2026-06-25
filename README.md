# 📈 Web App Quét Tín Hiệu Kỹ Thuật & Tối Ưu Hóa Danh Mục Đầu Tư

Ứng dụng trực quan hóa và cấu hình chiến lược đầu tư chứng khoán Việt Nam, sử dụng dữ liệu thực tế thời gian thực (Real-time) từ thư viện `vnstock`.

## 🌟 Tính năng chính
- **Bộ lọc cổ phiếu theo Tín hiệu:** Tự động tính toán chỉ báo và chấm điểm cổ phiếu dựa trên Xu hướng ($MA$), Động lượng dòng tiền ($RSI$) và Động năng ($Momentum$).
- **Tùy biến phân bổ vốn:** Cho phép người dùng linh hoạt chọn lựa giữa chiến lược Phân bổ đều hoặc Chiến lược 80-20 dựa trên thứ hạng điểm tín hiệu.
- **Backtest thực tế:** Trực quan hóa đường cong tăng trưởng tài sản tích lũy và đo lường hiệu quả qua Hệ số Sharpe ($Sharpe Ratio$).

## 🚀 Hướng dẫn khởi chạy dưới máy local
1. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
streamlit run app.py