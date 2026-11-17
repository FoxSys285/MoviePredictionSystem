# YÊU CẦU: XÂY DỰNG LẠI TOÀN BỘ MODULE THỰC NGHIỆM MACHINE LEARNING

## 1. Mục tiêu

Đập đi xây lại toàn bộ phần UI và chức năng thực nghiệm Machine Learning của dự án MovieLens 25M.

Không chỉ redesign UI mà phải kiểm tra, xây dựng lại và hoàn thiện cả quy trình thực nghiệm thực tế. Các thuật toán phải thực sự được huấn luyện, dự đoán, đánh giá và so sánh trên dữ liệu MovieLens 25M, không sử dụng dữ liệu giả lập để hiển thị kết quả.

Trước khi triển khai:
- Kiểm tra kiến trúc và công nghệ hiện tại của dự án.
- Kiểm tra cách xử lý dữ liệu MovieLens 25M.
- Kiểm tra các thuật toán và chức năng thực nghiệm hiện có.
- Xác định những phần cần giữ lại, chỉnh sửa hoặc xây dựng mới.
- Không tự ý phá vỡ các chức năng khác ngoài phạm vi thực nghiệm.

## 2. Yêu cầu UI

Thiết kế lại giao diện theo phong cách Modern Premium, chuyên nghiệp, phù hợp với một hệ thống nghiên cứu và thực nghiệm Machine Learning.

### Layout chính

Sử dụng Sidebar để điều hướng giữa các nhóm thuật toán:

1. Regression (Rating Prediction)
2. Classification
3. Clustering
4. Association Rules
5. Dimensionality Reduction
6. Reinforcement Learning

Khi chọn từng nhóm, hiển thị các nội dung:

- Giới thiệu mục tiêu thực nghiệm.
- Danh sách thuật toán thuộc nhóm.
- Các thông số cài đặt chung.
- Bảng các tiêu chí đánh giá và mô tả
- Các thông số và thống kê của Dataset đang sử dụng.
- Khu vực cấu hình và chạy thực nghiệm.
- Kết quả đánh giá, biểu đồ và bảng so sánh.

Sử dụng layout thống nhất giữa các nhóm nhưng nội dung cấu hình, metric và biểu đồ phải phù hợp với từng bài toán.

## 3. Thông số thực nghiệm chung

Áp dụng thống nhất:

- Random Seed: 42.
- Train/Validation/Test: 70/15/15.
- Cross Validation: 3-Fold đối với Regression và Classification.

Quy trình chia dữ liệu:
- Giữ riêng Test Set (15%), không sử dụng trong quá trình tuning.
- Thực hiện 3-Fold Cross Validation trên Train Set (70%).
- Validation Set (15%) dùng để lựa chọn cấu hình và Early Stopping khi cần.
- Sau khi chọn cấu hình, huấn luyện lại trên Train + Validation và đánh giá cuối cùng trên Test.

Các thông số Dataset cần hiển thị:
- Tổng số Ratings.
- Tổng số Users.
- Tổng số Movies.
- Tổng số Genres.
- Rating Min/Max/Mean.
- Tỷ lệ dữ liệu thưa (Sparsity) của ma trận User-Item.
- Số lượng bản ghi của Train, Validation và Test.

Không tính toán lại dữ liệu không cần thiết mỗi khi chuyển tab. Tái sử dụng kết quả tiền xử lý khi phù hợp.

## 4. Các nhóm thực nghiệm

### 4.1. Regression – Rating Prediction

Mục tiêu: Dự đoán rating của người dùng đối với phim trong khoảng 0.5–5.0 sao.

Thuật toán:
- Linear Regression.
- XGBoost Regressor.
- SVD (Collaborative Filtering).
- NCF (Neural Collaborative Filtering).

Tiêu chí:
- RMSE (↓).
- MAE (↓).
- Training Time (↓).
- Inference Time (↓).

Yêu cầu bổ sung:
- Báo cáo Mean ± Standard Deviation qua 3-Fold.
- NCF phải có Learning Curve thể hiện Train Loss và Validation Loss theo Epoch.
- Có Early Stopping để hạn chế Overfitting.
- Lưu lại cấu hình tốt nhất của từng thuật toán.

Kết quả cuối cùng:
Bảng so sánh 4 mô hình và nhận xét mô hình phù hợp nhất để triển khai dựa trên độ chính xác và chi phí tính toán.

### 4.2. Classification – Phân loại mức độ yêu thích

Mục tiêu: Phân loại rating thành hai nhãn:

- 0.5 ≤ rating < 3.5: Không thích (0).
- 3.5 ≤ rating ≤ 5.0: Thích (1).

Thuật toán:
- Logistic Regression.
- Random Forest Classifier.

Tiêu chí:
- Accuracy (↑).
- F1-score (↑).
- AUC-ROC (↑).
- Training Time (↓).
- Inference Time (↓).

Yêu cầu:
- Sử dụng Binary Classification.
- AUC-ROC tính từ xác suất dự đoán.
- Báo cáo Precision, Recall và Confusion Matrix bổ sung.
- Kiểm tra sự mất cân bằng giữa hai lớp.
- Báo cáo kết quả 3-Fold Mean ± Std.

### 4.3. Clustering – Phân cụm người dùng

Mục tiêu: Phân cụm người dùng dựa trên sở thích các thể loại phim.

Quy trình:
1. Xây dựng vector rating trung bình theo từng Genre cho mỗi User.
2. Xử lý dữ liệu thiếu và người dùng có quá ít ratings.
3. Chuẩn hóa vector đặc trưng.
4. Sử dụng dữ liệu sau chuẩn hóa để phân cụm.

Thuật toán:
- K-Means.
- DBSCAN.

Tiêu chí:
- Silhouette Score (↑).
- Davies-Bouldin Index (↓).
- Training Time (↓).
- Inference Time (↓).

Yêu cầu:
- K-Means: Thử nghiệm nhiều giá trị K.
- DBSCAN: Thử nghiệm eps và min_samples.
- Báo cáo số lượng cụm và tỷ lệ Noise.
- PCA 2D để trực quan hóa kết quả phân cụm.
- Phân tích đặc trưng Genre của từng cụm.
- Đối với DBSCAN, phải xác định rõ phương pháp xử lý và đo thời gian gán cụm cho dữ liệu mới vì thuật toán gốc không hỗ trợ predict chuẩn.

### 4.4. Association Rules – Khai phá luật kết hợp

Mục tiêu: Tìm các quy luật phim thường được người dùng yêu thích cùng nhau.

Thuật toán:
- Apriori.

Quy trình:
1. Xây dựng Transactions từ lịch sử đánh giá.
2. Chỉ xem các phim có rating từ 4.0 trở lên là phim được yêu thích.
3. Áp dụng Apriori để khai phá Frequent Itemsets.
4. Sinh Association Rules.

Tiêu chí:
- Support.
- Confidence.
- Lift.
- Execution Time.

Yêu cầu:
- Cho phép cấu hình min_support và min_confidence.
- Hiển thị danh sách luật kết hợp.
- Sắp xếp theo Lift, Confidence và Support.
- Hiển thị số lượng luật được sinh ra.
- Phân tích các luật có ý nghĩa trong gợi ý phim.

### 4.5. Dimensionality Reduction – Giảm chiều dữ liệu

Mục tiêu: Giảm số chiều dữ liệu nhưng giữ lại tối đa thông tin quan trọng.

Thuật toán:
- SVD.
- PCA.

Tiêu chí:
- Reconstruction Error (↓).
- Explained Variance (↑).
- Training Time (↓).
- Số chiều sau khi giảm.

Yêu cầu:
- Cho phép cấu hình số lượng Components.
- Hiển thị Explained Variance theo số Components.
- Trực quan hóa dữ liệu trước và sau giảm chiều.
- So sánh Reconstruction Error.
- Phân biệt rõ SVD và PCA, không nhầm lẫn SVD giảm chiều với SVD Collaborative Filtering ở Regression.

### 4.6. Reinforcement Learning – Sequential Recommendation

Mục tiêu: Xây dựng mô hình gợi ý phim tuần tự dựa trên phản hồi của người dùng.

Thuật toán:
- Q-Learning.

Baseline:
- Random Recommendation.
- Popularity-based Recommendation.

Tiêu chí:
- Cumulative Reward (↑).
- Average Reward (↑).
- Training Time (↓).

Yêu cầu:
- Xây dựng môi trường mô phỏng từ lịch sử MovieLens.
- Xác định State, Action, Reward và Policy.
- Cho phép cấu hình Learning Rate, Discount Factor và Exploration Rate.
- So sánh Q-Learning với hai Baseline trên cùng môi trường và điều kiện đánh giá.
- Hiển thị Reward theo Episode.
- Ghi rõ giới hạn của đánh giá offline/simulation.

## 5. Yêu cầu chức năng thực nghiệm chung

Mỗi nhóm phải hỗ trợ:

1. Chọn một hoặc nhiều thuật toán để thực nghiệm.
2. Cấu hình Hyperparameters.
3. Chạy riêng từng thuật toán hoặc chạy so sánh nhiều thuật toán.
4. Theo dõi trạng thái thực nghiệm.
5. Hiển thị kết quả sau khi hoàn thành.
6. Lưu lịch sử thực nghiệm.
7. Xem lại kết quả của các lần chạy trước.
8. Xuất kết quả CSV/Excel.
9. Hiển thị biểu đồ so sánh và bảng tổng hợp.

Không được sử dụng kết quả giả lập hoặc số liệu hardcode để thay thế kết quả thực nghiệm thật.

## 6. Đo lường hiệu năng

Đối với các thuật toán phù hợp, cần ghi nhận:

- Training Time.
- Inference Time.
- Model Size (nếu có thể).

Thống nhất môi trường phần cứng, cách đo và đơn vị thời gian.

Đối với những thuật toán không có cơ chế Inference chuẩn, cần ghi rõ phương pháp đo thay vì tạo ra số liệu không có ý nghĩa.

## 7. Yêu cầu tổng kết và lựa chọn mô hình

Mỗi nhóm phải có bảng tổng hợp kết quả, nhận xét và xác định thuật toán phù hợp nhất dựa trên các tiêu chí riêng của nhóm.

Đặc biệt, Regression phải có bảng tổng hợp cuối cùng so sánh:

- Linear Regression.
- XGBoost.
- SVD.
- NCF.

Đưa ra nhận xét về:
- Độ chính xác dự đoán.
- Khả năng tổng quát hóa.
- Thời gian huấn luyện.
- Thời gian dự đoán.
- Chi phí tài nguyên.
- Mức độ phù hợp để triển khai thực tế.

Không mặc định Deep Learning luôn tốt hơn Machine Learning truyền thống. Quyết định cuối cùng phải dựa trên kết quả thực nghiệm thực tế.

## 8. Yêu cầu triển khai

- Ưu tiên tái sử dụng công nghệ và kiến trúc hiện có.
- Tách biệt UI, xử lý dữ liệu, huấn luyện, đánh giá và lưu kết quả.
- Không thực hiện tác vụ huấn luyện nặng trực tiếp trên request UI.
- Có cơ chế theo dõi trạng thái và lỗi khi chạy thực nghiệm.
- Tránh xử lý ma trận User-Item dạng Dense gây tràn RAM.
- Không làm mất dữ liệu hoặc chức năng hiện tại ngoài phạm vi được yêu cầu.

## 9. Quy trình làm việc

Thực hiện theo từng giai đoạn:

1. Phân tích source code và kiến trúc hiện tại.
2. Đề xuất kiến trúc mới và kế hoạch thay đổi.
3. Thiết kế lại UI tổng thể.
4. Xây dựng lại pipeline tiền xử lý và chia dữ liệu.
5. Triển khai từng nhóm thuật toán.
6. Xây dựng hệ thống đánh giá và lưu kết quả.
7. Tích hợp UI với thực nghiệm thực tế.
8. Kiểm thử từng nhóm và kiểm tra tính chính xác của metric.
9. Hoàn thiện tài liệu hướng dẫn sử dụng.

Sau mỗi giai đoạn, kiểm tra và đảm bảo các chức năng đã hoàn thành hoạt động đúng trước khi chuyển sang giai đoạn tiếp theo.

Ưu tiên tính chính xác của thực nghiệm, khả năng tái lập kết quả và chất lượng kiến trúc hơn việc chỉ hoàn thiện giao diện đẹp.