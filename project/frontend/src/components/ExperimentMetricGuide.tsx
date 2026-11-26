import type { LabGroupId } from "../lib/lab";

const criteria = {
  rating: [
    { name: "RMSE", direction: "Càng thấp càng tốt", description: "Đo sai số dự đoán rating, phạt mạnh hơn với các dự đoán lệch nhiều so với rating thực tế." },
    { name: "MAE", direction: "Càng thấp càng tốt", description: "Cho biết rating dự đoán lệch trung bình bao nhiêu sao so với rating thực tế." },
  ],
  classification: [
    { name: "Accuracy", direction: "Càng cao càng tốt", description: "Tỷ lệ dự đoán đúng trong ba nhãn: Không thích, Bình thường và Thích." },
    { name: "F1-macro", direction: "Càng cao càng tốt", description: "Trung bình F1 của ba lớp, cân bằng Precision và Recall với trọng số như nhau cho mỗi lớp." },
    { name: "AUC-ROC", direction: "Càng cao càng tốt", description: "Đo khả năng phân biệt mỗi lớp với các lớp còn lại từ xác suất dự đoán, lấy trung bình theo One-vs-Rest." },
  ],
};

export function ExperimentMetricGuide({ groupId }: { groupId: LabGroupId }) {
  if (groupId !== "rating" && groupId !== "classification") return null;
  return (
    <section className="lab-section lab-metric-guide" aria-labelledby="lab-metric-guide-title" data-testid="lab-metric-guide">
      <div className="lab-metric-guide-heading">
        <h2 id="lab-metric-guide-title">Các tiêu chí đánh giá</h2>
        <p>{groupId === "rating"
          ? "Hồi quy: đánh giá rating dự đoán trên thang 0,5–5,0 sao."
          : "Phân loại: Không thích (<2,5), Bình thường (2,5–3,5), Thích (>3,5)."}</p>
      </div>
      <div className="lab-metric-guide-grid">
        {criteria[groupId].map((metric) => (
          <article className="lab-metric-guide-card" key={metric.name}>
            <h3>{metric.name}</h3>
            <p>{metric.description}</p>
            <span>{metric.direction}</span>
          </article>
        ))}
      </div>
    </section>
  );
}
