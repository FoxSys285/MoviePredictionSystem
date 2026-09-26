# CHƯƠNG 2. DỰ KIẾN CẤU TRÚC ĐỒ ÁN VÀ KẾ HOẠCH

## 2.1. Dự kiến cấu trúc các nội dung nghiên cứu, thực hiện trong đồ án

**Chương 1: Tổng quan**

Chương 1 giới thiệu bài toán gợi ý phim từ lịch sử đánh giá của người dùng. Phần mở đầu nêu khó khăn khi dữ liệu tương tác thưa, nhiều người dùng hoặc phim có ít lịch sử, và khi hệ thống cần tạo danh sách phim gợi ý từ các điểm dự đoán. Tiếp đó, chương tóm tắt các hướng nghiên cứu liên quan đến phân rã ma trận, học từ đặc trưng và Neural Collaborative Filtering; làm rõ đóng góp của đồ án là xây dựng một quy trình so sánh có thể kiểm tra lại trên MovieLens 25M. Cuối chương trình bày mục tiêu, đối tượng, phạm vi dữ liệu và những giới hạn của việc đánh giá ngoại tuyến. Nội dung này tương ứng với bản **Chương 1 – Mở đầu** nhóm đã soạn.

**Chương 2: Cơ sở lý thuyết**

Chương 2 cung cấp kiến thức cần thiết để hiểu các mô hình và kết quả ở những chương sau. Trước hết là cách biểu diễn dữ liệu người dùng–phim, sự khác nhau giữa điểm đánh giá đã quan sát và ô chưa có dữ liệu, cùng hai trường hợp người dùng mới và phim mới. Chương giải thích nguyên lý của baseline điểm trung bình, hồi quy tuyến tính, XGBoost, lọc cộng tác bằng KNN, phân rã ma trận/SVD và NCF; chỉ rõ đầu vào, đầu ra và điều kiện áp dụng của từng phương pháp. Các kỹ thuật phân loại, giảm chiều, phân cụm và khai phá luật kết hợp được trình bày ở mức cần thiết cho các thí nghiệm bổ sung. Phần cuối giới thiệu RMSE, MAE, các chỉ số gợi ý Top-N và các chỉ số riêng của từng bài toán. Các công trình, định nghĩa và công thức sử dụng trong chương sẽ có trích dẫn nguồn.

**Chương 3: Phương pháp đề xuất**

Chương 3 mô tả quy trình mà nhóm áp dụng cho MovieLens 25M. Quy trình gồm kiểm tra và làm sạch dữ liệu, chia tập theo thời gian, tạo đặc trưng từ train, huấn luyện các mô hình, dự đoán và đánh giá. Chương trình bày cụ thể tỷ lệ chia mục tiêu 70/15/15, năm lần kiểm chứng tiến theo thời gian trong train, cách lưu ID bản ghi để mọi mô hình dùng đúng cùng dữ liệu và quy tắc ngăn thông tin từ validation/test đi vào huấn luyện. Với các mô hình cần ID đã xuất hiện trong train, chương nêu cách báo cáo riêng nhóm có đủ lịch sử và cách dự phòng cho người dùng/phim mới. Sau cùng là quy trình tạo danh sách Top-N, loại phim người dùng đã chọn và tiêu chí chọn mô hình cho web demo dựa trên chất lượng dự đoán cùng thời gian xử lý.

## 2.2. Kế hoạch thực hiện dự kiến

Đồ án được tổ chức trong bốn tuần theo thứ tự **thu thập dữ liệu → chia tập → huấn luyện → dự đoán → đánh giá**. Việc huấn luyện và đánh giá trên train/validation có thể lặp lại khi cần điều chỉnh tham số; tập test chỉ được sử dụng sau khi nhóm thống nhất cấu hình cuối.

| Thời gian | Công việc trọng tâm | Sản phẩm cần có |
|---|---|---|
| Tuần 1 | Kiểm tra MovieLens 25M, khám phá dữ liệu, làm sạch và chia tập theo thời gian; thống nhất định dạng dữ liệu bàn giao. | Bộ dữ liệu đã chia, biểu đồ mô tả, quy tắc tiền xử lý và giao thức đánh giá chung. |
| Tuần 2 | Huấn luyện baseline và phiên bản đầu của Linear Regression, XGBoost, SVD, NCF; thử nghiệm các bài toán bổ sung và bản web demo ban đầu. | Mã huấn luyện, kết quả validation đầu tiên, bản demo nhận phim người dùng đã chọn. |
| Tuần 3 | Tìm tham số, chạy năm lần kiểm chứng, kiểm tra rò rỉ dữ liệu và so sánh chất lượng với chi phí tính toán. | Bảng so sánh validation, log thực nghiệm, mô hình và quy tắc dự phòng được chọn cho demo. |
| Tuần 4 | Khóa cấu hình, đánh giá trên test, hoàn thiện web, báo cáo và slide. | Kết quả test cuối cùng, web demo chạy được, báo cáo có thể đối chiếu với file kết quả. |

Phân công và sản phẩm chi tiết của từng thành viên được theo dõi trong các file `tasks/TV1.md` đến `tasks/TV5.md`; khi ghép báo cáo, nhóm sẽ cập nhật kế hoạch này theo phần việc và kết quả thực tế.
