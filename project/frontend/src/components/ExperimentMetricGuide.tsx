export function ExperimentMetricGuide() {
  return (
    <section className="lab-section lab-metric-guide" aria-labelledby="lab-metric-guide-title">
      <div className="lab-metric-guide-heading">
        <h2 id="lab-metric-guide-title">Tiêu chí đánh giá</h2>
        <p>Đánh giá sai số dự đoán rating trên thang 0,5–5,0 sao.</p>
      </div>
      <div className="lab-metric-guide-grid">
        <article className="lab-metric-guide-card">
          <h3>RMSE</h3>
          <p>Phạt nhiều hơn những dự đoán lệch xa rating thực tế.</p>
          <span>Càng thấp càng tốt</span>
        </article>
        <article className="lab-metric-guide-card">
          <h3>MAE</h3>
          <p>Cho biết dự đoán lệch trung bình bao nhiêu sao.</p>
          <span>Càng thấp càng tốt</span>
        </article>
      </div>
    </section>
  );
}
