import { useCallback, useState } from "react";
import { BarChart3, ChevronDown, Layers3, Sparkles } from "lucide-react";
import { Link, useSearchParams } from "react-router-dom";
import { PageHero } from "../components/ui";
import { ExperimentDataset } from "../components/ExperimentDataset";
import { ExperimentProtocol } from "../components/ExperimentProtocol";
import { ExperimentResults } from "../components/ExperimentResults";
import { ExperimentRunner } from "../components/ExperimentRunner";
import type { Dataset } from "../lib/api";
import { usePreferences } from "../lib/preferences";
import "../experiments.css";

const sections = [
  { id: "baseline", label: "Mô hình cơ sở", icon: Layers3 },
  { id: "improved", label: "Mô hình cải tiến", icon: Sparkles },
  { id: "evaluation", label: "So sánh và đánh giá kết quả thực nghiệm", icon: BarChart3 },
] as const;

const algorithms = [
  { id: "linear", name: "Linear Regression" },
  { id: "xgboost", name: "XGBoost" },
  { id: "svd", name: "SVD" },
  { id: "ncf", name: "NCF" },
] as const;

type ModelSection = "baseline" | "improved";
type AlgorithmId = typeof algorithms[number]["id"];

export function Experiments({ dataset, datasetLoading }: { dataset: Dataset | null; datasetLoading: boolean }) {
  const { t } = usePreferences();
  const [searchParams] = useSearchParams();
  const selected = sections.find((section) => section.id === searchParams.get("section")) ?? sections[0];
  const [baselineAlgorithms, setBaselineAlgorithms] = useState<AlgorithmId[]>([]);
  const [improvedAlgorithm, setImprovedAlgorithm] = useState<AlgorithmId>("svd");
  const modelSection = selected.id === "evaluation" ? null : selected.id;
  const [running, setRunning] = useState<Record<ModelSection, boolean>>({ baseline: false, improved: false });
  const onBusyChange = useCallback((variant: ModelSection, busy: boolean) => {
    setRunning((previous) => previous[variant] === busy ? previous : { ...previous, [variant]: busy });
  }, []);

  function toggleAlgorithm(algorithm: AlgorithmId) {
    setBaselineAlgorithms((current) => current.includes(algorithm)
      ? current.filter((id) => id !== algorithm) : [...current, algorithm]);
  }

  return <>
    <PageHero title={t("THỰC NGHIỆM")} subtitle={t("Khu vực thực nghiệm mô hình")} />
    <main id="main" className="container experiments-workspace">
      <aside className="experiments-sidebar" aria-labelledby="experiments-navigation-title">
        <h2 id="experiments-navigation-title">{t("Nội dung thực nghiệm")}</h2>
        <nav aria-label={t("Điều hướng thực nghiệm")}>
          {sections.map(({ id, label, icon: Icon }) => <Link
            key={id}
            to={`?section=${id}`}
            className={selected.id === id ? "is-active" : undefined}
            aria-current={selected.id === id ? "page" : undefined}
          >
            <Icon size={20} aria-hidden="true" />
            <span>{t(label)}</span>
          </Link>)}
        </nav>
      </aside>
      <section className="experiments-content" aria-labelledby="experiments-section-title">
        <h2 id="experiments-section-title">{t(selected.label)}</h2>
        {modelSection ? <div key={modelSection}>
        {modelSection === "improved" ? <div className="experiment-improved-setup">
          <p className="experiment-improved-intro">{t("Lựa chọn thuật toán và thiết lập các phương án cải tiến để đánh giá hiệu quả.")}</p>
          <div className="experiment-improved-algorithm">
            <label htmlFor="improved-algorithm">{t("Thuật toán cần cải tiến")}</label>
            <div className="experiment-improved-select">
              <select id="improved-algorithm" value={improvedAlgorithm} disabled={running.improved}
                onChange={(event) => setImprovedAlgorithm(event.target.value as AlgorithmId)}>
                {algorithms.map((algorithm) => <option key={algorithm.id} value={algorithm.id}>{algorithm.name}</option>)}
              </select>
              <ChevronDown size={19} aria-hidden="true" />
            </div>
          </div>
          <section className="experiment-improvement-plans" aria-labelledby="improvement-plans-title">
            <div className="experiment-improvement-plans-heading">
              <h3 id="improvement-plans-title">{t("Các phương án cải tiến")}</h3>
              <span>{t("{count} phương án", { count: 0 })}</span>
            </div>
            <div className="experiment-improvement-empty" role="status">
              <p>{t("Chưa có phương án cải tiến cho {algorithm}.", {
                algorithm: algorithms.find((algorithm) => algorithm.id === improvedAlgorithm)?.name,
              })}</p>
              <p>{t("Các phương án sẽ được bổ sung sau.")}</p>
            </div>
          </section>
        </div> : <fieldset className="experiment-algorithm-picker" disabled={running.baseline}>
          <legend>{t("Chọn thuật toán")}</legend>
          <div className="experiment-algorithm-grid">
            {algorithms.map((algorithm) => {
              const checked = baselineAlgorithms.includes(algorithm.id);
              return <label
                key={algorithm.id}
                className={`experiment-algorithm-option ${checked ? "is-selected" : ""}`}
                title={algorithm.name}
              >
                <input
                  type="checkbox"
                  name={`${modelSection}-algorithms`}
                  value={algorithm.id}
                  checked={checked}
                  onChange={() => toggleAlgorithm(algorithm.id)}
                />
                <span>{algorithm.name}</span>
              </label>;
            })}
          </div>
        </fieldset>}
        <ExperimentDataset dataset={dataset} loading={datasetLoading} />
        <ExperimentProtocol />
        <ExperimentRunner variant={modelSection}
          algorithms={modelSection === "baseline" ? baselineAlgorithms : [improvedAlgorithm]}
          disabledReason={modelSection === "improved" ? "Chưa có phương án cải tiến để thực nghiệm." : undefined}
          onBusyChange={onBusyChange} />
        </div> : <ExperimentResults />}
      </section>
    </main>
  </>;
}
