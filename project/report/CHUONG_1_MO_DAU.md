# CHƯƠNG 1. MỞ ĐẦU

> Bản nháp để đưa vào Word. Các mục mô tả **kế hoạch và mục tiêu dự kiến**; sau khi chạy thực nghiệm, nhóm cần cập nhật phạm vi dữ liệu thực dùng, mô hình triển khai và kết quả thực tế. Thay ký hiệu trích dẫn `[1]`–`[5]` theo định dạng tài liệu tham khảo mà giảng viên yêu cầu.

## 1.1. Đặt vấn đề

Số lượng phim trên các nền tảng trực tuyến ngày càng lớn, khiến người xem khó tìm được nội dung phù hợp với sở thích cá nhân. Một hệ thống gợi ý có thể hỗ trợ quá trình này bằng cách khai thác lịch sử đánh giá để dự đoán mức độ yêu thích của người dùng đối với các phim chưa xem. Tuy nhiên, dữ liệu đánh giá thường thưa: mỗi người chỉ chấm điểm một phần nhỏ trong số phim có sẵn. Hệ thống cũng cần xử lý những người dùng hoặc phim có ít tương tác, đồng thời đưa ra gợi ý đủ nhanh để sử dụng trong một ứng dụng web.

MovieLens là bộ dữ liệu thường được sử dụng để nghiên cứu hệ thống gợi ý phim. Bản MovieLens 25M cung cấp 25.000.095 lượt đánh giá từ 162.541 người dùng đối với 62.423 phim, cùng thông tin thời gian của từng lượt đánh giá [1], [5]. Quy mô này phù hợp để khảo sát cả chất lượng dự đoán lẫn chi phí tính toán của các phương pháp khác nhau. Từ đó, đồ án đặt ra câu hỏi: **trong cùng một quy trình dữ liệu và đánh giá, mô hình nào đạt sự cân bằng hợp lý giữa độ chính xác, thời gian huấn luyện, độ trễ dự đoán và khả năng tích hợp vào web demo?**

Đồ án tập trung vào dự đoán điểm đánh giá và tạo danh sách phim gợi ý Top-N. Các phương pháp hồi quy, phân rã ma trận và học sâu được đặt trong cùng một giao thức thực nghiệm. Một số kỹ thuật phân loại, phân cụm, luật kết hợp và giảm chiều được sử dụng để phân tích thêm dữ liệu và hỗ trợ nhận xét về hành vi người dùng.

## 1.2. Tình hình nghiên cứu và tính mới của đồ án

### 1.2.1. Nghiên cứu liên quan

Nghiên cứu về hệ thống gợi ý thường xuất phát từ ý tưởng sử dụng đánh giá của nhiều người để suy đoán sở thích của một người dùng đối với một phim. Harper và Konstan mô tả lịch sử, đặc điểm và giới hạn của các bộ dữ liệu MovieLens, đồng thời nhấn mạnh vai trò của dữ liệu đánh giá trong nghiên cứu gợi ý [1]. Trong nhóm phương pháp lọc cộng tác, Koren, Bell và Volinsky trình bày cách phân rã ma trận để học các yếu tố tiềm ẩn của người dùng và sản phẩm, qua đó dự đoán các điểm đánh giá chưa quan sát [2]. Đây là cơ sở để đồ án xây dựng mô hình SVD và đối chiếu với các baseline đơn giản.

Ngoài lọc cộng tác, bài toán dự đoán rating có thể được xây dựng như một bài toán học có giám sát từ các đặc trưng người dùng, phim và lịch sử tương tác. XGBoost của Chen và Guestrin là một hệ thống tăng cường cây quyết định có thiết kế hướng tới dữ liệu quy mô lớn [3]. Trong đồ án, Linear Regression đóng vai trò mô hình dễ diễn giải, còn XGBoost đại diện cho phương pháp có khả năng học các quan hệ phi tuyến từ đặc trưng được xây dựng.

He và cộng sự đề xuất Neural Collaborative Filtering (NCF), thay phép tương tác đơn giản giữa hai vector tiềm ẩn bằng một mạng nơ-ron để học quan hệ người dùng–sản phẩm [4]. Công trình gốc tập trung vào **phản hồi ngầm**; đồ án này điều chỉnh đầu ra và hàm mất mát của NCF để dự đoán **điểm đánh giá tường minh** của MovieLens 25M. Vì vậy, kết quả trong công trình gốc không thể được xem là kết quả trực tiếp cho bài toán rating của đồ án; hiệu quả của NCF cần được kiểm chứng thực nghiệm trên cùng dữ liệu với Linear, XGBoost và SVD.

### 1.2.2. Tính mới của đồ án

Đồ án không đề xuất một thuật toán gợi ý hoàn toàn mới. Điểm khác biệt trong phạm vi thực hiện là **xây dựng và kiểm tra một quy trình so sánh thống nhất** giữa mô hình hồi quy tuyến tính, tăng cường cây, phân rã ma trận và NCF trên MovieLens 25M. Nhóm dự kiến chia dữ liệu theo thời gian, dùng cùng các lần kiểm chứng cho bài toán dự đoán rating, cố định hạt giống ngẫu nhiên và ghi lại cấu hình từng lần chạy. Cách tổ chức này giúp nhận xét về mô hình dựa trên các phép đo có thể đối chiếu được.

Bên cạnh RMSE và MAE, đồ án xem xét chất lượng danh sách Top-N, thời gian huấn luyện, độ trễ suy luận và mức độ bao phủ đối với người dùng/phim ít tương tác. Kết quả so sánh sẽ được dùng để lựa chọn mô hình cho web demo, thay vì chọn trước một thuật toán chỉ dựa trên mô tả lý thuyết. Với người dùng mới chưa có ID trong dữ liệu huấn luyện, web demo cần một quy tắc gợi ý ban đầu và phải thể hiện rõ khi quy tắc này được sử dụng.

## 1.3. Mục tiêu nghiên cứu

### 1.3.1. Mục tiêu tổng quát

Xây dựng và đánh giá một hệ thống gợi ý phim cá nhân hóa từ dữ liệu MovieLens 25M; so sánh các mô hình dự đoán điểm đánh giá theo chất lượng và chi phí tính toán; từ kết quả đó chọn một phương án phù hợp để triển khai thành web demo gợi ý phim Top-N.

### 1.3.2. Mục tiêu nghiên cứu cụ thể

1. Thu thập, kiểm tra và mô tả dữ liệu MovieLens 25M; xử lý bản ghi không hợp lệ, tương tác ít và dữ liệu chưa quan sát theo quy tắc rõ ràng.
2. Chia dữ liệu thành train/validation/test theo thứ tự thời gian với tỷ lệ mục tiêu 70/15/15; tổ chức năm lần kiểm chứng tiến theo thời gian trong phần train.
3. Xây dựng baseline điểm đánh giá trung bình và các mô hình Linear Regression, XGBoost, SVD, NCF để dự đoán rating; tìm tham số phù hợp mà không sử dụng tập test để lựa chọn.
4. Thực hiện các phân tích bổ sung: phân loại mức độ yêu thích, phân cụm người dùng/phim, khai phá luật kết hợp và giảm chiều dữ liệu. Q-Learning là hướng mở rộng nếu nguồn lực cho phép.
5. Đánh giá các mô hình bằng các chỉ số phù hợp: RMSE, MAE, chỉ số gợi ý Top-N, thời gian huấn luyện, độ trễ suy luận; bổ sung Accuracy, F1, AUC-ROC, Silhouette, Davies–Bouldin, Support, Confidence và Lift cho các bài toán tương ứng.
6. Xây dựng web demo để người dùng chọn và chấm điểm một số phim đã xem, sau đó nhận danh sách phim gợi ý; mô tả cách xử lý khi người dùng hoặc phim chưa xuất hiện trong dữ liệu huấn luyện.

## 1.4. Đối tượng và quy mô nghiên cứu

### 1.4.1. Phạm vi

Đồ án sử dụng bộ dữ liệu MovieLens 25M do GroupLens công bố. Dữ liệu chính là các lượt đánh giá theo thang từ 0,5 đến 5 sao, có thời điểm đánh giá; thông tin tên và thể loại phim được dùng để hiển thị và có thể tạo đặc trưng bổ sung [5]. Quy mô gốc gồm 25.000.095 lượt đánh giá, 162.541 người dùng và 62.423 phim. Các số liệu thực dùng sau tiền xử lý sẽ được báo cáo riêng; nếu tài nguyên máy không cho phép huấn luyện mọi mô hình trên toàn bộ dữ liệu, nhóm sẽ dùng một tập con được chọn theo quy tắc cố định và áp dụng **cùng tập con** cho các mô hình cần so sánh.

Phạm vi đánh giá chính là **thực nghiệm ngoại tuyến** trên dữ liệu lịch sử. Web demo minh họa cách tạo gợi ý cho người dùng, chưa phải hệ thống vận hành với người dùng thực. MovieLens 25M không có thông tin nhân khẩu học và chỉ bao gồm những người dùng đã đánh giá ít nhất 20 phim; do đó, kết quả ngoại tuyến không thể tự chứng minh hiệu quả đối với mọi người dùng hoàn toàn mới [5].

### 1.4.2. Đối tượng

Đối tượng nghiên cứu là quan hệ giữa **người dùng, phim, điểm đánh giá và thời gian đánh giá**. Từ các tương tác đã quan sát, nhóm tìm hiểu khả năng dự đoán điểm cho cặp người dùng–phim chưa được đánh giá, khả năng xếp hạng phim để gợi ý Top-N và các nhóm sở thích thể hiện qua đặc trưng hoặc vector tiềm ẩn. Đối với phân loại, nhóm quy ước rating từ 4 sao trở lên là tín hiệu yêu thích; đối với luật kết hợp, mỗi người dùng được biểu diễn bằng tập phim họ đã đánh giá tích cực trong dữ liệu huấn luyện.

## 1.5. Nội dung và kỹ thuật nghiên cứu

### 1.5.1. Nội dung nghiên cứu

Nội dung đồ án được triển khai theo năm giai đoạn:

1. **Thu thập và khám phá dữ liệu:** đọc dữ liệu MovieLens 25M, thống kê quy mô, phân bố rating, số lượt tương tác theo người dùng/phim và mức độ thưa của ma trận tương tác.
2. **Tiền xử lý và chia tập:** kiểm tra dữ liệu không hợp lệ, xác định ngưỡng lọc từ train, lưu tương tác ở dạng thưa; chia train/validation/test theo thời gian và ghi nhận tỷ lệ người dùng/phim chưa có trong train.
3. **Huấn luyện mô hình:** xây dựng baseline, Linear, XGBoost, SVD và NCF; thực hiện tìm tham số. Các thí nghiệm Logistic Regression, Random Forest, K-Means, DBSCAN, PCA/TruncatedSVD và Apriori được tổ chức như các nội dung phân tích bổ sung.
4. **Dự đoán và gợi ý:** dự đoán điểm rating, xếp hạng các phim ứng viên, loại phim đã xem và tạo danh sách Top-N. Xây dựng giao diện web để người dùng nhập sở thích ban đầu và xem kết quả.
5. **Đánh giá và lựa chọn:** so sánh các mô hình trên cùng giao thức; lập bảng kết quả, phân tích sai số, thời gian chạy và khả năng xử lý trường hợp ít dữ liệu; chọn mô hình phù hợp cho demo.

### 1.5.2. Kỹ thuật nghiên cứu

Nhóm dự kiến sử dụng Python và các thư viện học máy phù hợp: scikit-learn cho hồi quy, phân loại, phân cụm và giảm chiều; XGBoost cho mô hình tăng cường cây; một phương pháp phân rã ma trận cho SVD; PyTorch cho NCF; mlxtend cho Apriori; Streamlit cho web demo. Cấu hình chạy, phiên bản thư viện và seed sẽ được ghi lại để hỗ trợ tái lập kết quả.

Dữ liệu được chia theo thứ tự `timestamp` với tỷ lệ mục tiêu 70% train, 15% validation và 15% test. Trong train, nhóm thực hiện năm lần kiểm chứng tiến theo thời gian; dùng GridSearchCV/RandomizedSearchCV cho các mô hình scikit-learn và Optuna cho NCF theo ngân sách tính toán được xác định trước. Nhóm chỉ dùng validation để kiểm tra và lựa chọn cấu hình cuối; tập test được đánh giá sau khi quy trình đã được khóa.

RMSE và MAE là hai chỉ số chính của dự đoán rating. Với gợi ý Top-N, nhóm dự kiến tính Precision@K, Recall@K và NDCG@K theo một quy tắc chọn phim ứng viên được công bố rõ. Các thí nghiệm phụ dùng chỉ số riêng phù hợp với từng bài toán. Thời gian huấn luyện và suy luận được đo trong cùng điều kiện phần cứng hoặc được ghi chú khi điều kiện khác nhau. Đường học train/validation của NCF được dùng để kiểm tra dấu hiệu quá khớp.

## Tài liệu tham khảo đề xuất cho Chương 1

[1] F. Maxwell Harper và Joseph A. Konstan (2015), [*The MovieLens Datasets: History and Context*](https://files.grouplens.org/papers/harper-tiis2015.pdf), ACM Transactions on Interactive Intelligent Systems, DOI: 10.1145/2827872.

[2] Yehuda Koren, Robert Bell và Chris Volinsky (2009), [*Matrix Factorization Techniques for Recommender Systems*](https://doi.org/10.1109/MC.2009.263), IEEE Computer, 42(8), tr. 30–37.

[3] Tianqi Chen và Carlos Guestrin (2016), [*XGBoost: A Scalable Tree Boosting System*](https://arxiv.org/abs/1603.02754), KDD 2016, DOI: 10.1145/2939672.2939785.

[4] Xiangnan He và cộng sự (2017), [*Neural Collaborative Filtering*](https://arxiv.org/abs/1708.05031), WWW 2017.

[5] GroupLens Research, [*MovieLens 25M Dataset README*](https://files.grouplens.org/datasets/movielens/ml-25m-README.html), mô tả dữ liệu và điều kiện sử dụng.
