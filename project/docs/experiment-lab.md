# Phòng thực nghiệm rút gọn

Mở `/experiments` và chọn nhóm ở sidebar. Trang chỉ giữ năm nhóm, 11 thuật toán
và đúng 17 chỉ số/biểu đồ:

| Nhóm | Các chỉ số | Số biểu đồ |
|---|---|---:|
| Hồi quy | MAE, MSE, RMSE, R² | 4 |
| Phân loại | Accuracy, Precision, Recall, F1 score | 4 |
| Phân cụm | Silhouette Score, CHI, DBI | 3 |
| Luật kết hợp | Support, Confidence, Lift | 3 |
| Giảm chiều dữ liệu | Explained Variance Ratio, Reconstruction Error, Downstream Model Performance | 3 |

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
  toàn cục 70/15/15, giữ cùng timestamp về một phía. Báo số user/rating của
  mẫu; đây không mặc định là toàn bộ MovieLens 25M.
- Hồi quy: Linear, XGBoost, SVD/MF có bias, NCF hồi quy; cùng rating cần dự
  đoán. MSE và R² được tính trực tiếp từ dự đoán. R² có thể âm, không bị ép
  về 0; N/A nếu rating thật không có phương sai. Linear/XGBoost dùng thêm
  metadata thể loại, được ghi rõ trong kết quả.
- Phân loại: Logistic và Random Forest. Precision/Recall/F1 dành cho lớp
  thích. Ngưỡng xác suất chọn theo F1 validation rồi khóa trước test.
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
