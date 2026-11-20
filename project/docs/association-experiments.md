# Association Rules: nguồn Train và nhật ký

Nhóm `association` / Apriori luôn dùng toàn bộ tập Train đã chuẩn bị. Backend chuẩn hóa `use_all_data=true` riêng nhóm này, kể cả khi yêu cầu cũ gửi `false`. `max_users` không giới hạn dữ liệu khai phá của nhóm này. Các nhóm khác giữ chế độ toàn bộ/lấy mẫu hiện có.

Các tệp trong `artifacts/data/`:

| Tệp | Vai trò với Apriori |
|---|---|
| `train.parquet` | Đọc toàn bộ ratings và users, thống kê số lượt thích để chọn phim ứng viên; khai phá itemset và sinh luật |
| `movies.parquet` | Tra tên phim cho các vế của luật |
| `validation.parquet`, `test.parquet` | Giữ riêng; được nạp để báo cáo nguồn và kích thước các tập, không dùng chọn phim hay sinh luật |
| `manifest.json` | Truy vết nguồn, tỷ lệ lấy mẫu tại bước chuẩn bị và kích thước các tập |

Không gộp cả ba tập vào Train. Dùng hết Train đã chuẩn bị cũng không tự biến một bộ dữ liệu nguồn đã lấy mẫu thành bộ đầy đủ: UI và log thông báo tỷ lệ nguồn nếu manifest có `user_sample_fraction < 1`. Không tự ghi đè hay chuẩn bị lại dữ liệu dùng chung.

Mỗi user Train là một transaction, gồm phim được chấm rating ≥ 4,0 thuộc catalog ứng viên. `max_items` giới hạn số phim ứng viên phổ biến nhất sau khi quét toàn bộ Train; đây không phải giới hạn số user hay lấy mẫu ratings. Transaction rỗng vẫn được giữ trong mẫu số support. Ma trận boolean chỉ tạo cho catalog ứng viên, tránh dựng bảng dense theo mọi phim đã quan sát.

UI của nhóm hiển thị nguồn tệp, số ratings/users Train sẵn sàng, phạm vi thực tế của từng lần chạy, lượt thích, phim ứng viên và transaction rỗng. Kết quả lịch sử giữ nguyên; phải chạy lại để có dữ liệu nguồn và nhật ký mới.

Nhật ký dùng cùng thành phần với hồi quy/phân loại: tự theo dõi dòng mới, cuộn xem log cũ, xem lại khi tải lại trang và tải JSONL. Các bước có log: nạp nguồn, quét toàn bộ Train, chọn phim ứng viên, tạo transaction, Apriori, số itemset theo độ dài, sinh luật, thống kê Train và lưu kết quả. Worker ghi heartbeat/RAM khi chạy lâu; lỗi và yêu cầu dừng được lưu, không công bố kết quả chạy dở.

Kết quả có `association_data` và `provenance.association_source` để truy vết số liệu và đường dẫn thật. Các chỉ số Support/Confidence/Lift hiện vẫn là trung bình các luật được lọc trên Train, chưa phải kiểm chứng chất lượng gợi ý trên dữ liệu tương lai.

Kiểm chứng trên dữ liệu hiện có ngày 17/11/2026: 17.500.066 ratings Train, 122.029 users, catalog mặc định 80 phim, 1.098 luật. Ma trận transaction 9.762.320 bytes. Kết quả có thể đổi khi nguồn hoặc ngưỡng khai phá đổi.

Kiểm tra:

```powershell
# Từ thư mục project
.\.venv\Scripts\python.exe -m pytest tests/test_experiment_association.py tests/test_experiment_logs.py tests/test_lab.py -q
cd frontend
npm run build
npm run test:association
```

`test:association` khởi chạy server kiểm thử cổng 8141 và tạo một lần chạy Apriori thật trên toàn bộ Train hiện có; kiểm tra việc chuyển từ nhóm đang lấy mẫu, log/lịch sử/tải xuống và UI desktop/mobile sáng/tối. Kết quả và ảnh chụp nằm trong `frontend/test-results/association-*`.

## Apriori phân nhóm có hỗ trợ giảm chiều

Chế độ mặc định vẫn là Apriori toàn cục ở trên. Khi chọn `enabled` hoặc `compare`, một biểu diễn sở thích tương đối + tần suất của mỗi user được fit từ Train, scale và nén bằng PCA/TruncatedSVD rồi gom user bằng K-Means. Cấu hình cho phép chọn danh sách `K` nhóm và số components cần thử. Mỗi nhóm khai phá Apriori trên các transaction Train của user thuộc nhóm đó.

Reducer không nhìn transaction/phim đã thích; transaction boolean và transaction rỗng giữ nguyên. Mẫu số support trong nhóm là tất cả user của nhóm, kể giao dịch rỗng; support toàn cục dùng mẫu số toàn Train. Hai chỉ số được lưu riêng, không so trực tiếp nếu mẫu số khác nhau.

Nhóm quá nhỏ hoặc không sinh được luật dùng luật toàn cục; nếu cũng không có luật khả dụng thì dùng Popularity. Validation chọn kích thước nhóm/representation bằng NDCG@10; Test chỉ báo cáo cuối. Cohort Validation/Test được lấy mẫu cố định, Train history làm context, phim positive ngoài catalog vẫn được tính vào eligible positives để báo candidate coverage.

Bảng kết quả/API lưu variant, k, số luật, support trong nhóm/toàn cục, nhóm fallback, coverage, NDCG Validation và fingerprint fit. Xuất CSV có các hàng luật theo nhóm và xếp hạng tương lai; XLSX thêm sheets `Grouped rules`, `Apriori groups` và `Future ranking`.
