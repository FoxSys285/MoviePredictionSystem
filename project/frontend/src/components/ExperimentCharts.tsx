import { usePreferences } from "../lib/preferences";

type Metrics = { rmse: number; mae: number };
export type ExperimentChartRow = {
  key: string; label: string; validation: Metrics; test: Metrics;
};

export function ExperimentCharts({ rows }: { rows: ExperimentChartRow[] }) {
  const { t, language } = usePreferences();
  const locale = language === "vi" ? "vi-VN" : "en-US";
  const baseline = rows.find((row) => row.key === "global_mean");
  const format = (value: number) => value.toLocaleString(locale, {
    minimumFractionDigits: 4, maximumFractionDigits: 4,
  });
  const formatPercent = (value: number) => value.toLocaleString(locale, {
    minimumFractionDigits: 1, maximumFractionDigits: 1,
  });

  const compareToBaseline = (value: number, baselineValue: number) => {
    const difference = baselineValue - value;
    const roundedDifference = Number(difference.toFixed(4));
    const percent = baselineValue === 0 ? 0 : Math.abs(difference / baselineValue) * 100;
    if (roundedDifference === 0) {
      return { label: "—", state: "is-even", description: t("Không đổi so với Global Mean") };
    }
    const isReduction = difference > 0;
    const amount = format(Math.abs(difference));
    const percentage = formatPercent(percent);
    return {
      label: `${isReduction ? "↓" : "↑"} ${amount} · ${percentage}%`,
      state: isReduction ? "is-lower" : "is-higher",
      description: t(isReduction
        ? "Giảm {amount} sao ({percent}%) so với Global Mean"
        : "Tăng {amount} sao ({percent}%) so với Global Mean", {
        amount,
        percent: percentage,
      }),
    };
  };

  return <div className="experiment-charts">
    {(["rmse", "mae"] as const).map((metric) => {
      const maximum = Math.max(...rows.flatMap((row) => [row.validation[metric], row.test[metric]]));
      const axisMax = Math.max(0.1, Math.ceil(maximum * 1.1 * 10) / 10);
      const title = `${t("Biểu đồ")} ${metric.toUpperCase()}`;
      return <figure className="experiment-chart" key={metric} aria-label={title}>
        <figcaption>
          <h3>{title}</h3>
          <p>{t("Giá trị càng thấp càng tốt.")} {t("So với Global Mean: chênh lệch và % cho từng tập.")}</p>
        </figcaption>
        <div className="experiment-chart-legend" aria-label={t("Chú giải")}>
          <span><i className="is-validation" aria-hidden="true" />Validation</span>
          <span><i className="is-test" aria-hidden="true" />Test</span>
        </div>
        <div className="experiment-chart-rows">
          {rows.map((row) => <div className="experiment-chart-row" key={row.key}>
            <div className="experiment-chart-label">{row.label}</div>
            <div className="experiment-chart-pair">
              {(["validation", "test"] as const).map((split) => {
                const value = row[split][metric];
                const label = split === "validation" ? "Validation" : "Test";
                const isBaseline = row.key === "global_mean";
                const comparison = baseline && !isBaseline
                  ? compareToBaseline(value, baseline[split][metric]) : null;
                return <div className="experiment-chart-series" key={split} role="group"
                  aria-label={`${row.label} · ${label} ${metric.toUpperCase()}: ${format(value)}${comparison ? `; ${comparison.description}` : ""}`}>
                  <div className="experiment-chart-track" title={`${label}: ${format(value)}`}>
                    <span className={`experiment-chart-bar is-${split}`} style={{ width: `${value / axisMax * 100}%` }} />
                  </div>
                  <span className="experiment-chart-value">{format(value)}</span>
                  <span className={`experiment-chart-delta ${comparison?.state || "is-baseline"}`}
                    title={comparison?.description || (isBaseline ? t("Baseline") : undefined)}>
                    {comparison?.label || (isBaseline ? t("Baseline") : "")}
                  </span>
                </div>;
              })}
            </div>
          </div>)}
        </div>
        <div className="experiment-chart-axis" aria-hidden="true"><div>
          {[0, 1, 2, 3, 4].map((tick) => <span key={tick}>{(axisMax * tick / 4).toLocaleString(locale, { maximumFractionDigits: 3 })}</span>)}
        </div></div>
        <p className="experiment-chart-unit">{t("Sai số (sao)")}</p>
      </figure>;
    })}
  </div>;
}
