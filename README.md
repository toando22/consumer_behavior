                                                    README - Phân Tích và Dự Đoán Hành Vi Khách Hàng
#Mô tả

Dự án này là một ứng dụng web được xây dựng bằng Streamlit để phân tích hành vi khách hàng từ tập dữ liệu CSV (ví dụ: marketing_campaign.csv).
Ứng dụng cho phép người dùng tải dữ liệu, tiền xử lý, phân cụm khách hàng (sử dụng KMeans và DBSCAN), trực quan hóa kết quả,
và dự đoán các chỉ số như tỷ lệ quay lại của khách hàng, số lượng khách hàng tiềm năng, cùng xu hướng trong tương lai (tùy chọn với Prophet).
Dự án được phát triển trong môi trường Anaconda với VS Code.

#Tính năng

Tải và xử lý dữ liệu: Tải file CSV và tự động làm sạch (xóa trùng lặp, giá trị thiếu, ngoại lai).
Phân cụm khách hàng: Sử dụng KMeans và DBSCAN với các tham số tùy chỉnh.
Trực quan hóa: Hiển thị histogram, countplot, heatmap, và scatter plot cho phân cụm.
Phân tích chuyên sâu: Tính tầm quan trọng đặc trưng bằng PCA.

#Dự đoán:

Tỷ lệ khách hàng quay lại dựa trên cột Response.
Số lượng khách hàng tiềm năng (thu nhập và chi tiêu cao hơn trung bình).
(Tùy chọn) Dự đoán xu hướng bằng Prophet.


#Xuất kết quả: Tải file CSV chứa kết quả phân cụm.

#Yêu cầu hệ thống

Hệ điều hành: Windows, macOS, hoặc Linux.
Python: Phiên bản 3.9 hoặc cao hơn.
Môi trường: Anaconda (được khuyến nghị).
Thư viện cần thiết:

streamlit
pandas
numpy
scipy
matplotlib
seaborn
scikit-learn
umap-learn
(Tùy chọn) prophet cho dự đoán xu hướng.



###Cài đặt###

Cài đặt Anaconda:

Tải từ anaconda.com và cài đặt.


Tạo môi trường ảo:

Mở Anaconda Prompt và chạy:
bashconda create -n consumer python=3.11
conda activate consumer



Cài đặt các thư viện:

Chạy các lệnh sau:
bashconda install streamlit pandas numpy scipy matplotlib seaborn scikit-learn
conda install -c conda-forge umap-learn
pip install prophet



Tải mã nguồn:

Clone hoặc tải file app.py từ repository (hoặc sao chép từ tài liệu này) vào thư mục C:\Consumer_Behavior.


Chạy ứng dụng:

Trong Terminal (VS Code hoặc Anaconda Prompt), chạy:
bashstreamlit run app.py

Mở trình duyệt tại http://localhost:8501.



###Cách sử dụng###

Tải file CSV: Nhấp vào "Tải file CSV" và chọn file (ví dụ: marketing_campaign.csv).
Chọn cột: Sử dụng menu thả xuống để chọn cột "Thu nhập", "Chi tiêu", và "Số lần mua sắm".
Điều chỉnh phân cụm: Chọn phương pháp (KMeans hoặc DBSCAN) và điều chỉnh số cụm/tham số.
Xem kết quả: Quan sát biểu đồ, bảng tóm tắt, và dự đoán.
Tải kết quả: Nhấp "Tải kết quả phân cụm" để lưu file CSV.

Ví dụ dữ liệu
Dữ liệu đầu vào nên chứa các cột như:

Income: Thu nhập của khách hàng.
Expenses: Tổng chi tiêu.
TotalNumPurchases: Số lần mua sắm.
Dt_Customer: Ngày tham gia (định dạng %d-%m-%Y).
Response: Trạng thái phản hồi (0 hoặc 1).

###Rắc rối thường gặp###

Lỗi ModuleNotFoundError: No module named 'prophet': Cài đặt pip install prophet hoặc comment đoạn code liên quan nếu không cần.
Lỗi định dạng file: Đảm bảo file CSV dùng tab \t (hoặc thay sep='\t' bằng sep=',' trong code).
Chậm khi xử lý dữ liệu lớn: Giảm số lượng hàng hoặc tối ưu code.

Đóng góp

Fork repository và tạo pull request.
Báo cáo lỗi hoặc đề xuất tính năng qua issues.

Giấy phép
Dự án này được phát hành dưới MIT License (nếu áp dụng).
Liên hệ

Tác giả: Do Duy Toan
Email: doduytoan2201@gmail.com
Ngày cập nhật: 14/08/2025