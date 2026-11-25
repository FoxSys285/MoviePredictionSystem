import type { LabGroup, LabPreset } from "../lib/lab";

export type ReductionChoice = "off" | "pca" | "truncated_svd";

const descriptions: Record<string, string> = {
  rating: "Nén đặc trưng tương tác cho Linear khi phù hợp.",
  classification: "Nén đặc trưng tương tác cho Logistic khi phù hợp.",
  clustering: "Nén hồ sơ người dùng trước khi phân cụm.",
  association: "Nén hồ sơ để phân nhóm trước Apriori.",
  reinforcement: "Có thể nén hồ sơ đầu vào của DQN.",
  pca: "Đối chiếu PCA và TruncatedSVD trên hồ sơ Train.",
};

const methodDescriptions = {
  pca: "Nén hồ sơ sở thích, giữ các hướng biến thiên chính.",
  truncated_svd: "Nén đặc trưng bằng phép chiếu không trừ trung bình.",
} as const;

export function ExperimentReductionGuide({
  group,
  preset,
  algorithms,
  choice,
  onChange,
  disabled,
}: {
  group: LabGroup;
  preset?: LabPreset;
  algorithms: string[];
  choice: ReductionChoice;
  onChange: (choice: ReductionChoice) => void;
  disabled: boolean;
}) {
  const methods = ["off", "pca", "truncated_svd"] as const;
  return (
    <section
      className="lab-section lab-reduction-guide"
      aria-labelledby="lab-reduction-guide-title"
      data-testid="lab-reduction-guide"
      data-group-id={group.id}
    >
      <div className="lab-reduction-heading">
        <div>
          <h2 id="lab-reduction-guide-title">Giảm chiều dữ liệu</h2>
          <p>{descriptions[group.id] ?? "Giảm chiều theo preset của nhóm."}</p>
        </div>
      </div>
      <div className="lab-reduction-grid" role="radiogroup" aria-label="Phương pháp giảm chiều">
        {methods.map((method) => {
          const appliedTo = algorithms.filter(
            (algorithm) => preset?.algorithms[algorithm]?.reduction?.method === method,
          );
          const selected = choice === method;
          const state = selected ? "Đã chọn" : "Chưa chọn";
          const dimension = appliedTo
            .map((algorithm) => `${group.algorithms.find((item) => item.id === algorithm)?.name ?? algorithm} · ${preset?.algorithms[algorithm]?.reduction?.components} chiều`)
            .join(", ");
          return (
            <article
              className={`lab-reduction-card${selected ? " active" : ""}`}
              key={method}
              data-testid={`reduction-card-${method}`}
              data-method={method}
              data-active={appliedTo.length > 0}
            >
              <div className="lab-reduction-card-heading">
                <h3>{method === "off" ? "Không giảm chiều" : method === "pca" ? "PCA" : "TruncatedSVD"}</h3>
                <span className={`lab-reduction-status${selected ? " active" : ""}`}>
                  {state}
                </span>
              </div>
              <p>{method === "off" ? "Dùng đặc trưng gốc, bỏ qua bước giảm chiều." : methodDescriptions[method]}</p>
              {appliedTo.length > 0 && <small>{dimension}</small>}
              {selected && method !== "off" && !appliedTo.length && <small>Không áp dụng cho các thuật toán đang chọn.</small>}
              <label className="lab-reduction-toggle">
                <input type="radio" name="reduction-choice" value={method} checked={selected}
                  disabled={disabled} onChange={() => onChange(method)}/>{" "}
                {method === "off" ? "Bỏ giảm chiều" : `Chọn ${method === "pca" ? "PCA" : "TruncatedSVD"}`}
              </label>
            </article>
          );
        })}
      </div>
    </section>
  );
}
