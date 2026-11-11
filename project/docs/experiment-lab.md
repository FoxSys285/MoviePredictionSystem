# Phòng thực nghiệm rút gọn — tài liệu lịch sử v2

> Module hiện tại đã chuyển sang pipeline v3 với sáu nhóm. Xem
> [hướng dẫn Experiment Studio](experiment-studio.md) và
> [kết quả xác nhận](experiment-studio-results.md). Phần dưới mô tả giao thức cũ,
> được giữ để diễn giải các báo cáo đã lưu trước khi nâng cấp.

Mở `/experiments` và chọn nhóm ở sidebar. Trang chỉ giữ năm nhóm, 11 thuật toán
và đúng 17 chỉ số/biểu đồ:

Xem [kết quả và hướng dẫn diễn giải ngưỡng](experimental-results.md) để biết
mỗi nhóm giải quyết yêu cầu nào, chỉ số nào dùng chọn mô hình và cách đọc
kết quả Validation/Test.

| Nhóm | Các chỉ số | Số biểu đồ |
|---|---|---:|
| Hồi quy | MAE, MSE, RMSE, R² | 4 |
| Phân loại | Accuracy, Precision, Recall, F1 score | 4 |
| Phân cụm | Silhouette Score, CHI, DBI | 3 |
| Luật kết hợp | Support, Confidence, Lift | 3 |
| Giảm chiều dữ liệu | Explained Variance Ratio, Reconstruction Error, Downstream Model Performance | 3 |

## Bài toán, dữ liệu đầu vào và cách đọc ngưỡng

| Nhóm thuật toán | Bài toán giải quyết | Dự đoán dựa trên gì | Các chỉ số đánh giá | Dùng các đặc trưng nào | Mốc diễn giải tham khảo |
|---|---|---|---|---|---|
| **Hồi quy**<br>Linear Regression, XGBoost Regressor, SVD/MF, NCF | Ước lượng rating dạng số (0,5–5 sao) cho một cặp user–phim. | Rating lịch sử trong train. Linear Regression/XGBoost học từ thống kê user/phim và thể loại; SVD/NCF học mức độ phù hợp tiềm ẩn giữa user và phim từ các rating. | **MAE, MSE, RMSE, R².** | Linear/XGBoost: rating trung bình của user/phim, log số rating, cờ user/phim chưa có lịch sử, thể loại phim one-hot, độ lệch rating của user theo thể loại và tương tác với thể loại phim. SVD/NCF: userId/movieId dùng làm khóa danh mục cùng rating quan sát được; SVD học bias user/phim, NCF học embedding và tương tác phi tuyến. | MAE ≤ **0,50 sao**: tốt; 0,50–0,75: chấp nhận được cho thử nghiệm ban đầu; >0,75: còn yếu theo mốc dự án. R² > 0 tốt hơn baseline dự đoán bằng trung bình; R² = 0 tương đương baseline; R² < 0 kém hơn. MSE/RMSE càng thấp càng tốt, cần so trên cùng tập. |
| **Phân loại**<br>Logistic Regression, Random Forest | Dự đoán rating thuộc lớp “thích” hay “không thích”; mặc định rating ≥ 4 là “thích”. | Đặc trưng lịch sử của user–phim; nhãn train được tạo từ rating theo ngưỡng thích. Ngưỡng xác suất dự đoán được chọn theo F1 trên validation rồi giữ nguyên khi chấm test. | **Accuracy, Precision, Recall, F1**; Precision/Recall/F1 là trung bình macro của hai lớp. | Rating trung bình của user/phim, log số rating, cờ user/phim chưa có lịch sử, thể loại phim one-hot, độ lệch rating của user theo thể loại và tương tác với thể loại phim. | Các chỉ số nằm trong [0, 1], thường cao hơn là tốt. Mốc vận hành của dự án: Precision ≥ **0,70** nếu cần dự đoán đáng tin; Recall ≥ **0,70** nếu ưu tiên tìm đủ phim thích; F1 ≥ **0,70** làm mốc cân bằng ban đầu. Accuracy cần so với baseline lớp đa số; không có ngưỡng chung cho mọi ứng dụng. |
| **Phân cụm**<br>K-Means, DBSCAN | Khám phá nhóm phim hoặc nhóm user có hồ sơ tương tự; không dự đoán nhãn đã biết. | Vector đặc trưng của từng phim hoặc user, chuẩn hóa bằng scaler chỉ fit trên train; K-Means/DBSCAN tìm cấu trúc cụm trong vector đó. | **Silhouette Score, CHI (Calinski–Harabasz), DBI (Davies–Bouldin).** | Nếu phân cụm phim: thể loại, rating trung bình, log(1 + số rating), độ lệch chuẩn rating. Nếu phân cụm user: tỷ lệ các thể loại trong lịch sử phim đã đánh giá, rating trung bình, log(1 + số rating), độ lệch chuẩn rating. | Silhouette ≤ **0,25**: chưa thấy cấu trúc cụm rõ; >0,25–0,50: yếu; >0,50–0,70: tương đối hợp lý; >0,70: mạnh. Đây là mốc kinh nghiệm. CHI cao hơn và DBI thấp hơn thường tốt hơn, nhưng không có ngưỡng tuyệt đối; cần so cùng dữ liệu/đặc trưng. Cần ít nhất hai cụm hợp lệ để tính chỉ số. |
| **Luật kết hợp**<br>Apriori, FP-Growth | Tìm các phim thường được cùng user đánh giá tích cực, biểu diễn thành luật như A → B. | Giao dịch của mỗi user là tập phim có rating đạt ngưỡng “thích”; Apriori/FP-Growth khai phá luật từ giao dịch train, sau đó đo cùng tập luật trên validation/test. | **Support, Confidence, Lift**, thường lấy trung bình trên tập luật. | Các movieId được user đánh giá tích cực (mặc định rating ≥ 4); giới hạn danh mục theo các phim phổ biến được chọn trong train. Không dùng vector thể loại trong nhóm này. | Lift > **1**: liên kết dương; =1: gần độc lập; <1: liên kết âm. Không có ngưỡng Support/Confidence phổ quát; đọc cùng số giao dịch và tần suất nền. Cấu hình hiện tại dùng min_support **0,08** và confidence train ≥ **0,30** để tạo luật — đây là ngưỡng khai phá, không phải chứng nhận luật “tốt”. |
| **Giảm chiều dữ liệu (PCA)**<br>PCA | Nén vector đặc trưng và kiểm tra việc giảm chiều ảnh hưởng ra sao đến dự đoán rating. PCA tự nó không dự đoán rating; Ridge được dùng sau PCA để đo hiệu quả dự đoán. | Đặc trưng user–phim được chuẩn hóa rồi chiếu lên các thành phần chính; Ridge học dự đoán rating từ vector đã chiếu. | **Explained Variance Ratio (EVR), Reconstruction Error, Downstream Model Performance (RMSE của Ridge).** | Cùng nhóm đặc trưng hồi quy dạng metadata: rating trung bình của user/phim, log số rating, cờ user/phim chưa có lịch sử, thể loại phim one-hot, độ lệch rating của user theo thể loại và tương tác với thể loại phim; sau đó chuẩn hóa và biến đổi PCA. | EVR thuộc [0, 1]; EVR ≥ **0,90** có thể là mục tiêu nếu cần giữ ít nhất 90% phương sai, nhưng không đảm bảo giữ thông tin dự đoán rating. Reconstruction Error thấp hơn là tốt hơn, không có mốc chung. Với RMSE, so Ridge sau PCA với Ridge không PCA trên cùng tập; thấp hơn hoặc gần baseline hơn là tốt. |

Các mốc trên là hướng dẫn thực hành của đề tài, không phải chuẩn đúng cho mọi bộ dữ liệu. Dùng **Validation** để chọn cấu hình; giữ nguyên lựa chọn rồi dùng **Test** để xác nhận. Với các chỉ số không có ngưỡng tuyệt đối, so sánh baseline trên cùng cách chia dữ liệu và cùng đặc trưng. Xem thêm [kết quả và hướng dẫn diễn giải ngưỡng](experimental-results.md).

Mỗi biểu đồ có hai chuỗi cột Validation và Test, có số trên từng cột. Hai tập
đánh giá cũng xuất hiện cạnh nhau trong cùng bảng; không có nút đổi tập.
Không có biểu đồ chi phí, top-N, bootstrap, ROC/PR, heatmap, đường học hoặc
Q-Learning trong giao diện này.

## Cách chạy

1. Đọc mô tả thuật toán và bảng tiêu chí của nhóm.
2. Chọn số user tối đa, seed và tham số phù hợp rồi chạy nhóm.
3. Xem đồng thời Validation/Test trong bảng và các biểu đồ. N/A là không đủ
   dữ liệu hoặc chỉ số không xác định, không được thay bằng 0.
4. Có thể xem lại lịch sử hoặc tải CSV/JSON của lần chạy.

Tác vụ chạy nền, có tiến độ và yêu cầu dừng. Chỉ chạy một nhóm tại một thời
điểm. Định dạng kết quả rút gọn có `schema_version=2`; kết quả cũ vẫn giữ trên
đĩa nhưng không trộn vào lịch sử của giao thức mới.

## Cách đo

- Dữ liệu: lấy mẫu trọn lịch sử user từ prepared data, chia theo thời gian
  70/15/15 trong từng user cho Hồi quy/Phân loại; các nhóm khác chia toàn cục.
  User cần ít nhất 10 rating và ba khoảng timestamp tách biệt. Giữ cùng
  timestamp về một phía. Thời gian giữa các user có thể chồng lấn; giao thức
  này đánh giá cá nhân hóa, không mô phỏng tương lai toàn cục. Báo số user/rating của
  mẫu; đây không mặc định là toàn bộ MovieLens 25M.
- Hồi quy: Linear, XGBoost, SVD/MF có bias, NCF hồi quy; cùng rating cần dự
  đoán. MSE và R² được tính trực tiếp từ dự đoán. R² có thể âm, không bị ép
  về 0; N/A nếu rating thật không có phương sai. Linear/XGBoost dùng thêm
  metadata thể loại, được ghi rõ trong kết quả.
- Phân loại: Logistic và Random Forest. Precision/Recall/F1 dùng trung bình
  macro hai lớp. Ngưỡng xác suất chọn theo F1 macro validation rồi khóa trước test.
  Không so trực tiếp với kết quả cũ chia toàn cục và F1 riêng lớp thích.
- Phân cụm: K-Means và DBSCAN chỉ fit trên vector train; scaler fit train.
  Tạo vector valid/test bằng cùng lược đồ đặc trưng. Với phim, giữ danh mục
  phim train đã chọn và chỉ đánh giá phim có tương tác trong từng tập giữ
  lại. Với user, đánh giá các user có tương tác trong từng tập. K-Means dùng
  predict; DBSCAN gán nhãn của core train gần nhất nếu khoảng cách ≤ eps.
  Các chỉ số tính trên điểm không nhiễu, cần ít nhất hai cụm hợp lệ. CHI là
  Calinski–Harabasz Index; DBI là Davies–Bouldin Index.
- Luật kết hợp: Apriori và FP-Growth khai phá cùng giao dịch train, cùng
  min_support, confidence ≥0,3, độ dài tối đa 2 phim. Giữ nguyên tập luật
  này để đo trên giao dịch valid/test với cùng danh mục phim. Bảng báo
  trung bình Support/Confidence/Lift trên các luật, loại giá trị không xác
  định khi mẫu số bằng 0. Hai thuật toán có thể cho số bằng nhau vì chúng
  khai phá cùng tập luật; không gọi đây là so sánh tốc độ.
- PCA: fit PCA/scaler bằng train; so các số chiều với cấu hình không PCA.
  EVR là tổng phương sai train giữ lại, dùng chung cho valid/test.
  Reconstruction Error là MSE tái tạo trên đặc trưng chuẩn hóa của từng
  tập. Downstream Model Performance dùng RMSE của cùng Ridge alpha=1.

## Kết quả và API

Kết quả nằm ở `artifacts/experiment-lab/<run-id>/job.json` và `result.json`.
Không thay mô hình gợi ý đang phục vụ ứng dụng hoặc `latest.json`. Cấu hình,
dấu vết mẫu, phiên bản thư viện vẫn được lưu trong JSON để tái lập.

Các endpoint dưới `/api/v1/experiment-lab`: `GET /catalog`, `GET /runs`,
`POST /runs`, `GET /runs/{id}`, `POST /runs/{id}/cancel`,
`GET /runs/{id}/export/csv` và `GET /runs/{id}/export/json`.

## Kiểm tra

`python -m pytest` kiểm tra thuật toán, chỉ số, chia tập và tác vụ nền.
`npm run build` kiểm tra TypeScript và build production.
`npm run test:experiments` chạy thật năm nhóm trên mẫu MovieLens đã chuẩn bị,
kiểm tra chính xác 4/4/3/3/3 biểu đồ, hai tập xuất hiện cùng lúc, nhãn giá trị
trên cột, xuất kết quả và giao diện mobile sáng/tối. Script dùng Chrome/Edge
headless và backend riêng cổng 8139, tự đóng sau khi xong.
