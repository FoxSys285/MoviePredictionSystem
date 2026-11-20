# Mốc tham khảo cho các tiêu chí thực nghiệm

Các mốc dưới đây là **mục tiêu khởi đầu do dự án lựa chọn**, không phải chuẩn
chứng nhận chất lượng cho mọi dataset. Giao diện cho phép sửa từng mốc, nhớ trên
trình duyệt và khôi phục mặc định. Chỉnh mốc chỉ đổi nhận xét cạnh kết quả; không
chạy lại training, không đổi metric và không chọn lại mô hình bằng Test.

## Mốc chất lượng và độ gọn

| Nhóm | Tiêu chí | Mốc mặc định | Cách diễn giải và lý do |
|---|---|---|---|
| Regression | RMSE | ≤1,00 sao | Mục tiêu sai số căn quân phương tối đa khoảng 1 sao trên thang 0,5–5. RMSE nhấn mạnh lỗi lớn; 0,80 đạt, 1,20 chưa đạt. Đây là mục tiêu ban đầu, cần đối chiếu baseline và CV ± Std. |
| Regression | MAE | ≤0,75 sao | Mục tiêu lệch tuyệt đối trung bình không quá 0,75 sao. MAE 0,50 nghĩa là sai lệch trung bình nửa sao; đọc cùng RMSE để thấy lỗi lớn. |
| Classification | Accuracy | ≥0,70 | Ít nhất 70% nhãn đúng; 0,80 đạt mốc, 0,69 chưa đạt. Nếu baseline luôn dự đoán lớp đa số Train đã đạt 0,85 trên tập đang đánh giá thì Accuracy 0,80 được đánh dấu cần thận trọng. |
| Classification | F1 lớp Thích | ≥0,70 | Mục tiêu cân bằng Precision/Recall của lớp Thích. F1 thấp có thể do gợi ý sai hoặc bỏ sót nhiều; F1 không phản ánh TN nên vẫn cần Confusion Matrix. |
| Classification | AUC-ROC | ≥0,70 | Mục tiêu phân biệt hai lớp tốt hơn mức không phân biệt 0,50. AUC không phải Accuracy và không chứng minh xác suất đã được hiệu chỉnh tốt. |
| Clustering | Silhouette | ≥0,50 | Heuristic cho mức tách biệt cụm tương đối rõ. Gần 0 gợi ý chồng lấn; giá trị âm có thể là gán cụm không phù hợp. Không có ngưỡng chất lượng phổ quát. |
| Clustering | Davies-Bouldin | ≤1,00 | Mục tiêu độ phân tán trong cụm nhỏ tương đối với khoảng cách cụm. DBI phụ thuộc biểu diễn, scaler, khoảng cách và số cụm, nên chỉ so cùng điều kiện. |
| Association | Support | ≥0,08 | Luật xuất hiện trong ít nhất 8% transactions Train. Đây là mốc độ phủ, không phải độ chính xác; tính cả transaction rỗng của user Train trong mẫu số. |
| Association | Confidence | ≥0,30 | Ít nhất 30% user có vế trái cũng có vế phải, khớp min_confidence ban đầu. Confidence cao có thể chỉ do phim vế phải phổ biến; cần xem Lift. |
| Association | Lift | >1,00 | Lift >1 biểu thị đồng xuất hiện dương; =1 không tăng so độc lập; <1 liên kết âm. Lift 1,20 tương ứng confidence cao hơn tần suất nền 20%, không phải độ chính xác 120%. |
| Reduction | Reconstruction MSE | ≤0,10 | Mục tiêu mất mát tái tạo nhỏ trên vector user–genre đã chuẩn hóa. Không dùng mốc này trên dữ liệu chưa chuẩn hóa hay coi đó là sai số dự đoán rating. |
| Reduction | Explained Variance | ≥0,90 | Giữ ít nhất 90% phương sai Train; TruncatedSVD dùng phương sai thực tế của thành phần chiếu chia cho tổng phương sai đầu vào đã scale, không dùng tỷ lệ năng lượng. Dưới mốc có thể thử thêm components; phương sai không đồng nghĩa độ chính xác tác vụ gợi ý. |
| Reduction | Components | ≤8 chiều | Ngân sách nén khởi đầu cho vector khoảng 19 genres. Chỉ hiện đánh giá tốt về độ gọn khi EVR và reconstruction error cũng đạt các mốc tương ứng. |
| RL | Cumulative Reward | Δ>0 so baseline | Δ = reward DQN − max(reward Random, reward Popularity) trên cùng split/episode/horizon. Cumulative reward phụ thuộc số action thực tế; đọc cùng reward/action và coverage. |
| RL | Average Reward/action | Δ>0 so baseline | So reward mỗi hành động với cùng hai baseline. Không lấy reward thô >0 làm bằng chứng DQN tốt hơn; baseline có thể tốt hơn dù reward DQN dương. |

Nhãn “Tốt theo mốc” chỉ đánh giá **một tiêu chí**, không kết luận toàn mô hình.
Giá trị N/A không được đổi thành 0 hoặc gắn nhãn tốt/xấu. Khi thay mốc tùy chỉnh,
quy tắc so sánh giữ nguyên: ≥ với accuracy/F1/AUC, ≤ với lỗi và ngân sách, > với
Lift và chênh lệch reward. Các ví dụ trong phần giải thích vẫn ghi rõ là mốc mặc định.

Silhouette/DBI của DBSCAN đang loại noise; một kết quả đẹp có thể chỉ đến từ số
ít user dễ phân cụm. Phải đọc cùng tỷ lệ noise, số cụm và đặc trưng genre.

Reward chỉ được so khi có cả Random và Popularity, cùng split và số episode.
Nếu thiếu baseline, giao diện ghi “Thiếu baseline”, không tự lấy 0 thay thế.
Ngay cả khi vượt baseline, reward offline chưa chứng minh hiệu quả online do
phản hồi thiếu được quy ước là 0, coverage hạn chế và không có propensity.

## Ngân sách hiệu năng

| Tiêu chí | Mốc khởi đầu | Ý nghĩa |
|---|---|---|
| Training Time / Execution Time | ≤60 giây | Ngân sách một lần fit cuối hoặc khai phá trên mẫu đã chọn; không phải tổng thời gian tuning/CV. DQN có đánh giá checkpoint Validation trong thời gian học. |
| Inference Time | ≤0,10 giây/batch | Ngân sách 100 ms cho **cả batch**, không phải mỗi rating. Chỉ so batch cùng số dòng, phần cứng, số luồng và phạm vi phép đo. |

Nhãn thời gian là “Trong ngân sách” / “Vượt ngân sách”, không phải dự đoán
tốt/xấu. Khi tăng mẫu hoặc thay máy, cần điều chỉnh ngân sách. Classification
chỉ đo predict_proba; Regression gồm predict và bước tạo đặc trưng trong predict.

## Mở rộng và thu gọn cấu hình

Phần “Cấu hình thông số” mặc định thu gọn. Nhấp hoặc dùng Enter/Space trên tiêu
đề để mở/đóng. Các input vẫn giữ giá trị khi thu gọn; nút chạy, chọn thuật toán,
lịch sử và tiến độ luôn hiện. Nếu input không hợp lệ đang bị ẩn, nhấn chạy sẽ tự
mở phần cấu hình để sửa.

## Nguồn định nghĩa

Các con số mục tiêu 0,70 / 0,50 / 0,10 / 60 giây là chính sách của dự án, không
phải ngưỡng tiêu chuẩn rút từ các tài liệu dưới đây. Nguồn dùng để xác minh ý
nghĩa của chỉ số:

- [scikit-learn: metrics và ảnh hưởng của mất cân bằng lớp](https://scikit-learn.org/stable/modules/model_evaluation.html).
- [scikit-learn: Silhouette có miền giá trị −1 đến 1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html).
- [scikit-learn: AUC-ROC từ xác suất hoặc điểm quyết định](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
- [mlxtend: Support, Confidence và Lift](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/).

Kiểm tra logic mốc bằng `npm run test:metric-references` trong `frontend`.
Kiểm tra tương tác thu gọn, giữ input, sửa/lưu/khôi phục mốc và chạy thật qua
`npm run test:experiments`.
