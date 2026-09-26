# CHƯƠNG 2. DỰ KIẾN CẤU TRÚC ĐỒ ÁN VÀ KẾ HOẠCH

## 2.1. Dự kiến cấu trúc các nội dung nghiên cứu, thực hiện trong đồ án

Ba chương đầu của báo cáo cuối dự kiến có nội dung như sau:

**Chương 1: Tổng quan**

Chương đầu giải thích vì sao nhóm chọn bài toán gợi ý phim. Người dùng chỉ đánh giá một phần rất nhỏ số phim có trong hệ thống; với người vừa tham gia, lịch sử này còn ít hơn. Từ vấn đề đó, nhóm giới thiệu MovieLens 25M, điểm qua những hướng dự đoán điểm đánh giá đã được nghiên cứu và xác định câu hỏi cần trả lời trong đồ án: mô hình nào phù hợp với dữ liệu, tài nguyên tính toán và web demo của nhóm. Phần cuối nêu mục tiêu, phạm vi nghiên cứu cùng giới hạn của việc đánh giá bằng dữ liệu có sẵn. Nội dung này đã được viết trong bản **Chương 1 – Mở đầu**.

**Chương 2: Cơ sở lý thuyết**

Phần lý thuyết bắt đầu từ ma trận người dùng–phim và ý nghĩa của một ô chưa có rating. Sau đó, nhóm trình bày cách các phương pháp được chọn xử lý bài toán: dự đoán từ đặc trưng bằng hồi quy, dự đoán từ lịch sử tương tác bằng KNN và phân rã ma trận, hoặc học quan hệ người dùng–phim bằng NCF. Mô hình điểm trung bình được giải thích như một mốc đối chiếu. Với mỗi phương pháp, chương nêu ý tưởng, dữ liệu đầu vào và trường hợp dễ gặp khó khăn, nhất là khi người dùng hoặc phim chưa xuất hiện trong train. Những kiến thức về phân loại, phân cụm, giảm chiều và luật kết hợp được đưa vào đúng phần thí nghiệm có sử dụng. Chương cũng giải thích các thước đo cần thiết để người đọc hiểu bảng kết quả ở phần sau; định nghĩa và công thức sẽ được dẫn nguồn.

**Chương 3: Phương pháp đề xuất**

Chương này mô tả các bước nhóm thực hiện trên MovieLens 25M để người khác có thể làm lại thí nghiệm. Sau khi kiểm tra và làm sạch dữ liệu, rating được sắp theo thời gian rồi chia thành train, validation và test với tỷ lệ mục tiêu 70/15/15. Trong train, nhóm tạo năm lần kiểm chứng tiến theo thời gian. Chương ghi rõ dữ liệu nào được dùng để tạo đặc trưng, huấn luyện và chọn tham số; mỗi dự đoán được gắn với ID bản ghi để đối chiếu đúng điểm thật. Do validation có nhiều người dùng chưa xuất hiện trong train, kết quả trên toàn tập và trên nhóm đã có đủ lịch sử sẽ được báo cáo riêng. Cuối chương là cách xử lý người dùng mới, tạo danh sách phim chưa xem và chọn mô hình cho demo dựa trên cả độ chính xác lẫn thời gian xử lý.

## 2.2. Kế hoạch thực hiện dự kiến

Nhóm dự kiến làm trong bốn tuần. Thứ tự chung là **thu thập dữ liệu, chia tập, huấn luyện, dự đoán và đánh giá**. Trong lúc thử mô hình, nhóm có thể quay lại bước huấn luyện sau khi xem kết quả validation; tập test được giữ cho lần đánh giá cuối.

**Tuần 1:** Kiểm tra MovieLens 25M, làm sạch rating và thống nhất cách chia dữ liệu. Kết thúc tuần, mọi thành viên cần đọc được cùng các tập train, validation, test và hiểu rằng cặp người dùng–phim chưa có rating là tương tác chưa được quan sát.

**Tuần 2:** Chạy baseline và các mô hình đầu tiên trên train, lấy dự đoán ở validation. Thành viên phụ trách web chuẩn bị giao diện chọn phim để kiểm tra luồng gợi ý, kể cả trường hợp người dùng mới.

**Tuần 3:** Điều chỉnh tham số bằng các lần kiểm chứng trong train, rà soát rò rỉ dữ liệu và tổng hợp kết quả validation. Nhóm chọn mô hình đưa vào demo sau khi đối chiếu sai số, số trường hợp dự đoán được và thời gian chạy.

**Tuần 4:** Giữ nguyên cấu hình đã chọn để đánh giá trên test. Các thành viên hoàn thiện phần báo cáo mình phụ trách; nhóm ghép báo cáo, kiểm tra lại số liệu và chạy thử web trước khi trình bày.
