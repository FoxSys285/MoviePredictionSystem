# Phòng thực nghiệm theo sáu nhóm thuật toán

Mở `/experiments`. Sidebar chia 12 thuật toán theo nhiệm vụ: dự đoán rating,
phân loại sở thích, phân cụm, luật kết hợp, PCA và Q-Learning. Nội dung mô tả,
tiêu chí, câu hỏi đánh giá và danh mục biểu đồ lấy từ `Mô tả.docx`, được lưu
trong `src/movie_rating/lab_catalog.json`. Bản Word có thể tải ngay từ sidebar.

## Cách sử dụng

1. Chọn nhóm trong sidebar và đọc mô tả các thuật toán, đầu ra và bảng tiêu chí.
2. Chọn số user, seed và tham số của nhóm. Mặc định lấy tối đa 150 user từ dữ
   liệu đã chuẩn bị. Với nhóm có ngẫu nhiên, chọn 2–3 seed nếu cần đo std.
3. Chạy toàn bộ thuật toán trong nhóm. Có tiến độ và nút dừng; tác vụ đang
   fit một mô hình sẽ dừng ở ranh giới bước kế tiếp. Máy chủ chạy một tác vụ
   tại một thời điểm để tránh chồng tài nguyên.
4. Đọc bảng kết quả, chuyển Validation/Test khi phù hợp, xem phạm vi mẫu và
   lưu ý. Chỉ số chưa đo hoặc không xác định là N/A.
5. Xem biểu đồ; bấm chú giải để ẩn/hiện chuỗi, di chuột để xem giá trị hoặc
   mở bảng số liệu. Tải CSV/JSON và chọn lại lần chạy trong lịch sử.

## Dữ liệu và tính tái lập

Phòng thực nghiệm yêu cầu `artifacts/data/{train,validation,test,movies}.parquet`.
Nó lấy mẫu user có seed, giữ toàn bộ lịch sử còn trong prepared data, rồi chia
lại theo mốc thời gian toàn cục 70/15/15, giữ các timestamp bằng nhau cùng phía.
Đây là mẫu của dữ liệu đã chuẩn bị, không mặc định là toàn bộ MovieLens 25M.
Mỗi báo cáo công bố số mẫu, mốc thời gian, dấu vết manifest/mẫu, cấu hình, seed,
phiên bản thư viện và hash mã nguồn.

Các lần chạy được lưu tại `artifacts/experiment-lab/<run-id>/job.json` và
`result.json`. Chúng không thay mô hình phục vụ gợi ý, `latest.json` hoặc các
artifact huấn luyện cũ. Nếu máy chủ dừng giữa chừng, lần chạy được hiển thị
gián đoạn sau khi khởi động lại; cần chạy lại để có kết quả hoàn chỉnh.

## Phạm vi các phép đo

- Rating: Linear Regression, XGBoost, SVD có bias, NCF hồi quy; thêm Global Mean
  và Bias Baseline. Có RMSE, MAE, thiên lệch, phân tầng rating/user, bootstrap
  ghép cặp theo user, đường học NCF và top-N trên tối đa 15 user phù hợp.
  Linear/XGBoost dùng thêm thể loại; báo cáo nêu rõ đây là khác biệt đầu vào
  so với SVD/NCF. Thống kê huấn luyện loại rating của chính mẫu khỏi đặc trưng.
- Phân loại: Logistic và Random Forest, thêm baseline lớp đa số. Ngưỡng nhãn
  rating và ngưỡng xác suất là hai khái niệm khác nhau; ngưỡng xác suất chọn
  bằng Macro-F1 validation rồi giữ cố định ở test. Có confusion matrix, PR,
  ROC, Precision/Recall/F1 và AP theo định nghĩa của scikit-learn.
- Phân cụm: cùng vector phim hoặc user từ train, cùng chuẩn hóa, K-Means và
  DBSCAN. Có điểm cụm, kích thước, tỷ lệ nhiễu, heatmap, phép chiếu 2D, khảo
  sát K/eps/min_samples. Có bảng chấm hai phương pháp trên tập không nhiễu
  chung. Không đủ hai cụm hợp lệ thì điểm hình học là N/A.
- Association: cùng tập user và top phim tích cực từ train; cùng support,
  confidence ≥0,3, độ dài tối đa 2 để giới hạn số kết quả trong thao tác web.
  Có kiểm tra tập phổ biến/support tương đương, số luật, thời gian khai phá,
  thời gian sinh luật, RSS và khảo sát quy mô/support; tối đa 100 luật hiện ở
  bảng và 400 luật ở scatter. Số lượng thống kê dùng toàn bộ kết quả khai phá.
- PCA: so k=1/2/4/8/16/hạng tối đa hợp lệ với baseline không PCA, giữ cùng
  Ridge alpha=1 sau giảm chiều. Có phương sai, lỗi tái tạo trên validation/test,
  sai số rating, thời gian pipeline và chiếu 2D. Quy tắc chọn trên validation
  công bố trong kết quả; PCA/scaler chỉ fit bằng train.
- Q-Learning: simulator công bố rõ, tối đa 12 phim train, trạng thái gồm nhóm
  hồ sơ/tập phim đã chọn/phim trước; horizon=5, gamma=0,9, alpha=0,2. Reward
  được ước lượng từ train và độ đa dạng thể loại, có nhiễu; không phải phản hồi
  thực cho hành động mới. So với Random/Popular/Greedy trên cùng 60 episode
  đánh giá có seed tách huấn luyện, policy Q dùng epsilon=0. Không suy từ
  simulator sang hiệu quả triển khai thật.

RAM là RSS tiến trình lấy mẫu trong thao tác, bao gồm bộ nhớ máy chủ; không
phải RAM riêng của mô hình. p50/p95 cho hồi quy/phân loại là lặp cùng truy vấn
sau warm-up; top-N và RL có phạm vi đo riêng ghi trong báo cáo. Thanh sai số
là std qua seed, không gọi đó là CI 95%. Biểu đồ chẩn đoán dùng seed đầu nếu
bảng kết quả đang tổng hợp nhiều seed.

## API

Các endpoint bắt đầu bằng `/api/v1/experiment-lab`:

| Endpoint | Nội dung |
|---|---|
| GET `/catalog` | Nhóm, mô tả, tiêu chí, danh mục biểu đồ, trạng thái dữ liệu |
| GET `/runs?group=rating` | Lịch sử, có thể lọc nhóm |
| POST `/runs` | Tạo tác vụ từ group và cấu hình đã kiểm tra giới hạn |
| GET `/runs/{id}` | Tiến độ, trạng thái và báo cáo hoàn tất |
| POST `/runs/{id}/cancel` | Yêu cầu dừng tại ranh giới bước tính toán |
| GET `/runs/{id}/export/csv` | Bảng kết quả |
| GET `/runs/{id}/export/json` | Toàn bộ số liệu, biểu đồ và provenance |
| GET `/document` | Tải Word mô tả |

## Kiểm tra

`python -m pytest` chạy các kiểm tra backend, gồm cả sáu bộ chạy, temporal split,
N/A, tương đương tập phổ biến, bootstrap, lưu/khôi phục, CSV/JSON và hủy tác vụ.
`npm run build` kiểm tra TypeScript và build production.
`npm run test:experiments` mở trình duyệt headless trên backend riêng cổng 8139,
chạy cả sáu nhóm bằng mẫu MovieLens thật đã chuẩn bị, lưu kết quả độc lập và
ảnh trong `frontend/test-results/`. Tác vụ này cần Chrome/Edge cài trên máy.
