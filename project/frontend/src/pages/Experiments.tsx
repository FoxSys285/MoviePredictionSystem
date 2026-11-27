import { useState } from "react";
import { BarChart3, Layers3, Sparkles } from "lucide-react";
import { Link, useSearchParams } from "react-router-dom";
import { PageHero } from "../components/ui";
import { ExperimentDataset } from "../components/ExperimentDataset";
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
  const [selections, setSelections] = useState<Record<ModelSection, AlgorithmId[]>>({
    baseline: [],
    improved: [],
  });
  const modelSection = selected.id === "evaluation" ? null : selected.id;
  const ActiveIcon = selected.icon;

  function toggleAlgorithm(section: ModelSection, algorithm: AlgorithmId) {
    setSelections((current) => ({
      ...current,
      [section]: current[section].includes(algorithm)
        ? current[section].filter((id) => id !== algorithm)
        : [...current[section], algorithm],
    }));
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
        {modelSection ? <><fieldset className="experiment-algorithm-picker">
          <legend>{t("Chọn thuật toán")}</legend>
          <div className="experiment-algorithm-grid">
            {algorithms.map((algorithm) => {
              const checked = selections[modelSection].includes(algorithm.id);
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
                  onChange={() => toggleAlgorithm(modelSection, algorithm.id)}
                />
                <span>{algorithm.name}</span>
              </label>;
            })}
          </div>
        </fieldset>
        <ExperimentDataset dataset={dataset} loading={datasetLoading} />
        </> : <div className="experiments-empty">
          <ActiveIcon size={36} aria-hidden="true" />
          <h3>{t("Chưa có nội dung")}</h3>
          <p>{t("Nội dung của mục này sẽ được bổ sung ở bước tiếp theo.")}</p>
        </div>}
      </section>
    </main>
  </>;
}
