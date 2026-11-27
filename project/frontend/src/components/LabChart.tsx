import { usePreferences } from "../lib/preferences";
import { metricNumber, type LabChart } from "../lib/lab";

const colors = ["#de81b0", "#55c6c0", "#b1a1ff", "#e8be71"];
const finite = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

export function LabChartFigure({ chart }: { chart: LabChart }) {
  const { language } = usePreferences();
  const categories = [...new Set(chart.series.flatMap((series) => series.points.map((point) => point.x)))];
  const values = chart.series.flatMap((series) => series.points.map((point) => point.y)).filter(finite);
  const baseline = chart.baseline && finite(chart.baseline.value) ? chart.baseline : undefined;
  if (baseline) values.push(baseline.value);
  if (!categories.length || !values.length) return null;

  const width = Math.max(760, categories.length * 132 + 100);
  const height = 370;
  const left = 82, right = 24, top = 52, bottom = 112;
  const minimum = Math.min(0, ...values), maximum = Math.max(0, ...values);
  const span = maximum - minimum || 1;
  const minY = minimum < 0 ? minimum - span * 0.2 : 0;
  const maxY = maximum + span * 0.25;
  const y = (value: number) => height - bottom - (value - minY) / (maxY - minY) * (height - top - bottom);
  const slot = (width - left - right) / categories.length;
  const barWidth = Math.min(52, slot * 0.7 / Math.max(1, chart.series.length));
  const number = (value: unknown) => metricNumber(value, language);

  return <article className="lab-chart" data-chart-kind="bar" data-metric={chart.metric_key}>
    <div className="lab-chart-heading"><span>{String(chart.index + 1).padStart(2, "0")}</span><small>{chart.split}</small></div>
    <h3>{chart.title}</h3>
    <div className="lab-plot-scroll"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={chart.title} className="lab-plot" style={{ minWidth: width, maxWidth: "none" }}>
      <title>{chart.title} · {chart.split}</title>
      {[0, 1, 2, 3, 4].map((index) => {
        const value = minY + (maxY - minY) * index / 4;
        return <g key={index}><line className="lab-gridline" x1={left} x2={width - right} y1={y(value)} y2={y(value)}/>
          <text x={left - 9} y={y(value) + 4} textAnchor="end">{number(value)}</text></g>;
      })}
      <line className="lab-zero-line" x1={left} x2={width - right} y1={y(0)} y2={y(0)}/>
      {categories.map((category, index) => {
        const x = left + (index + 0.5) * slot;
        const label = String(category);
        return <text key={label} x={x} y={height - bottom + 23} textAnchor="end" transform={`rotate(-28 ${x} ${height - bottom + 23})`}>
          <title>{label}</title>{label.length > 22 ? `${label.slice(0, 20)}…` : label}
        </text>;
      })}
      <text x={(width + left - right) / 2} y={height - 9} textAnchor="middle">{chart.x_label}</text>
      {chart.series.map((series, seriesIndex) => <g key={series.label} data-series={series.label}>
        {series.points.map((point, pointIndex) => {
          const categoryIndex = categories.indexOf(point.x);
          const groupWidth = barWidth * chart.series.length + 8 * (chart.series.length - 1);
          const x = left + (categoryIndex + 0.5) * slot - groupWidth / 2 + seriesIndex * (barWidth + 8);
          const center = x + barWidth / 2;
          const value = finite(point.y) ? point.y : undefined;
          const labelY = value === undefined ? y(0) - 10 : y(value) + (value < 0 ? 18 : -10);
          return <g key={pointIndex}>{value !== undefined && <rect x={x} y={Math.min(y(0), y(value))} width={barWidth}
            height={Math.max(0.8, Math.abs(y(value) - y(0)))} fill={colors[seriesIndex % colors.length]} rx={2}/>}
            <text className={value !== undefined ? "lab-bar-value" : "lab-bar-missing"} x={center} y={labelY} textAnchor="middle"
              transform={`rotate(-38 ${center} ${labelY})`}>{number(point.y)}</text>
          </g>;
        })}
      </g>)}
      {baseline && <g className="lab-chart-baseline" data-baseline-value={baseline.value}>
        <line x1={left} x2={width - right} y1={y(baseline.value)} y2={y(baseline.value)} stroke="var(--lab-baseline, #f4bf67)" strokeWidth={2} strokeDasharray="8 5"/>
        <text x={width - right} y={y(baseline.value) - 9} textAnchor="end">{baseline.label}: {number(baseline.value)}</text>
      </g>}
    </svg></div>
    <div className="lab-legend lab-static-legend">{chart.series.map((series, index) => <span key={series.label}><i style={{ background: colors[index % colors.length] }}/>{series.label}</span>)}
      {baseline && <span className="lab-baseline-legend"><i/>{baseline.label}: {number(baseline.value)}</span>}
    </div>
  </article>;
}
