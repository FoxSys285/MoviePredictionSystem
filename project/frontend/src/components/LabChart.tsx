import { useMemo, useState } from "react";
import { usePreferences } from "../lib/preferences";
import type { ChartGuide, LabChart, LabPoint } from "../lib/lab";

const colors = ["#de81b0", "#55c6c0", "#d9b763", "#9e9cf2", "#86c985", "#f0a36e", "#75b6eb", "#e39ac6"];
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

export function LabChartFigure({ chart, guide }: { chart: LabChart; guide?: ChartGuide }) {
  const { language, t } = usePreferences();
  const [hidden, setHidden] = useState<string[]>([]);
  const number = (v: number) => v.toLocaleString(language === "en" ? "en-US" : "vi-VN", { maximumFractionDigits: 4 });
  const all = chart.series.flatMap(s => s.points);
  const available = all.filter(p => finite(p.y));
  const visible = chart.series.filter(s => !hidden.includes(s.label));
  const width = 680, height = chart.kind === "bar" ? 380 : 310;
  const left = 68, right = 22, top = 28, bottom = chart.kind === "bar" ? 112 : 58;
  const categories = [...new Set(all.map(p => p.x))];
  const numericX = chart.kind !== "bar" && categories.every(finite);
  const xs = categories.filter(finite);
  const initialMinX = numericX ? Math.min(...xs) : 0;
  const initialMaxX = numericX ? Math.max(...xs) : Math.max(1, categories.length - 1);
  const minX = initialMinX === initialMaxX ? initialMinX-.5 : initialMinX;
  const maxX = initialMinX === initialMaxX ? initialMaxX+.5 : initialMaxX;
  const minY = Math.min(0, ...available.map(p => p.y! - (p.error || 0)));
  const maxY = Math.max(.001, ...available.map(p => p.y! + (p.error || 0))) * 1.1;
  const scaleX = (v: number | string) => left + ((numericX ? Number(v) - minX : categories.indexOf(v)) / Math.max(.0001, maxX - minX)) * (width-left-right);
  const scaleY = (v: number) => height-bottom-(v-minY)/Math.max(.0001, maxY-minY)*(height-top-bottom);
  const ticks = useMemo(() => numericX ? Array.from({ length: 5 }, (_, i) => minX+(maxX-minX)*i/4) : categories, [numericX, minX, maxX, categories.join("|")]);
  const tooltip = (label: string, p: LabPoint) => `${label} · ${p.label || p.x}: ${finite(p.y) ? number(p.y) : "N/A"}${p.count !== undefined ? ` · n=${p.count}` : ""}${p.error !== undefined ? ` ± ${number(p.error)} (std)` : ""}${p.value !== undefined ? ` · ${chart.color_value || "Giá trị"}=${number(p.value)}` : ""}`;
  const matrix = chart.matrix || [];
  const columns = chart.column_labels || [];
  const values = matrix.flat();
  const heatMax = Math.max(.001, ...values.map(v => Math.abs(v)));
  const valueMin = Math.min(...available.map(p => p.value || 0)), valueMax = Math.max(1, ...available.map(p => p.value || 0));
  return <article className="lab-chart" data-chart-kind={chart.kind}>
    <div className="lab-chart-heading"><span>{chart.index >= 0 ? `0${chart.index + 1}` : chart.index === -2 ? "TOP-N" : "CHI PHÍ"}</span><small>{chart.split}</small></div>
    <h3>{chart.title}</h3>
    {guide && <p className="lab-chart-question">{guide.question}</p>}
    {chart.kind === "heatmap" ? matrix.length && columns.length ? <div className="lab-heatmap-scroll"><table className="lab-heatmap"><caption>{chart.y_label} × {chart.x_label}</caption><thead><tr><th scope="col">{chart.y_label}</th>{columns.map((c, i) => <th key={i} scope="col">{c}</th>)}</tr></thead><tbody>{matrix.map((row, i) => <tr key={i}><th scope="row">{chart.row_labels?.[i] || i}</th>{row.map((v, j) => { const total = row.reduce((a, b) => a+b, 0); return <td key={j} style={{ backgroundColor: `rgba(180,100,158,${.1+.65*Math.abs(v)/heatMax})` }} title={`${chart.row_labels?.[i]} × ${columns[j]}: ${number(v)}`}><b>{number(v)}</b>{chart.normalize_rows && <small>{total ? number(v/total*100) : "0"}%</small>}</td>; })}</tr>)}</tbody></table></div> : <p className="lab-empty-chart">{t("Không đủ dữ liệu hợp lệ để vẽ biểu đồ.")}</p> : available.length ? <>
      <div className="lab-plot-scroll"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={chart.title} className="lab-plot">
        <title>{chart.title}</title><text x={left} y={15} className="lab-axis-title">{chart.y_label}</text>
        {Array.from({ length: 5 }, (_, i) => { const value = minY+(maxY-minY)*i/4; return <g key={i}><line className="lab-gridline" x1={left} x2={width-right} y1={scaleY(value)} y2={scaleY(value)} /><text x={left-9} y={scaleY(value)+4} textAnchor="end">{number(value)}</text></g>; })}
        {ticks.map((tick, i) => <text key={i} x={chart.kind === "bar" ? left+(i+.5)*(width-left-right)/categories.length : scaleX(tick)} y={height-bottom+20} textAnchor={chart.kind === "bar" ? "end" : "middle"} transform={chart.kind === "bar" ? `rotate(-32 ${left+(i+.5)*(width-left-right)/categories.length} ${height-bottom+20})` : undefined}><title>{tick}</title>{typeof tick === "number" ? number(tick) : tick.length > 19 ? tick.slice(0,17)+"…" : tick}</text>)}
        <text x={(width+left-right)/2} y={height-8} textAnchor="middle" className="lab-axis-title">{chart.x_label}</text>
        {visible.map(s => {
          const ci = chart.series.findIndex(v => v.label === s.label);
          const valid = s.points.filter(p => finite(p.y));
          return <g key={s.label}>
            {chart.kind === "line" && <polyline fill="none" stroke={colors[ci%colors.length]} strokeWidth={2.3} points={valid.map(p => `${scaleX(p.x)},${scaleY(p.y!)}`).join(" ")} />}
            {valid.map((p, pi) => {
              const slot = (width-left-right)/Math.max(1, categories.length);
              const barWidth = Math.min(60, slot*.76/Math.max(1, visible.length));
              const x = chart.kind === "bar" ? left+categories.indexOf(p.x)*slot+slot*.12+visible.indexOf(s)*barWidth : scaleX(p.x);
              const color = chart.color_value && p.value !== undefined ? `hsl(${45+275*(p.value-valueMin)/Math.max(.001, valueMax-valueMin)} 65% 65%)` : colors[ci%colors.length];
              return <g key={pi}>
                {chart.kind === "bar" ? <rect x={x} y={Math.min(scaleY(0), scaleY(p.y!))} width={barWidth-2} height={Math.max(1, Math.abs(scaleY(p.y!)-scaleY(0)))} rx={2} fill={color}><title>{tooltip(s.label,p)}</title></rect> : <circle cx={x} cy={scaleY(p.y!)} r={chart.kind === "scatter" ? 3.7 : 3} fill={color} opacity={chart.kind === "scatter" ? .75 : 1}><title>{tooltip(s.label,p)}</title></circle>}
                {p.error !== undefined && <g stroke={color}><line x1={x+(chart.kind === "bar" ? barWidth/2 : 0)} x2={x+(chart.kind === "bar" ? barWidth/2 : 0)} y1={scaleY(p.y! - p.error)} y2={scaleY(p.y! + p.error)} /><title>± {number(p.error)} std qua seed</title></g>}
              </g>;
            })}
          </g>;
        })}
      </svg></div>
      <div className="lab-legend">{chart.series.map((s,i) => <button key={s.label} type="button" className={hidden.includes(s.label) ? "is-hidden" : ""} aria-pressed={!hidden.includes(s.label)} onClick={() => setHidden(v => v.includes(s.label) ? v.filter(x => x !== s.label) : [...v,s.label])}><i style={{ background: colors[i%colors.length] }} />{s.label}</button>)}</div>
      {chart.color_value && <p className="lab-chart-note">{chart.color_value}: {number(valueMin)} → {number(valueMax)} · {t("Di chuột lên điểm để xem chi tiết.")}</p>}
    </> : <p className="lab-empty-chart">{t("Không đủ dữ liệu hợp lệ để vẽ biểu đồ.")}</p>}
    {chart.note && <p className="lab-chart-note">{chart.note}</p>}
    {guide && <details className="lab-chart-details"><summary>{t("Cách đọc biểu đồ")}</summary><p>{guide.axes}</p></details>}
    {!!available.length && <details className="lab-chart-details"><summary>{t("Xem số liệu biểu đồ")}</summary><div className="lab-table-scroll lab-chart-data"><table><thead><tr><th>{t("Chuỗi dữ liệu")}</th><th>{chart.x_label}</th><th>{chart.y_label}</th><th>{t("Ghi chú")}</th></tr></thead><tbody>{chart.series.flatMap(s => s.points.filter(p => finite(p.y)).map((p,i) => <tr key={`${s.label}-${i}`}><th scope="row">{s.label}</th><td>{p.label || p.x}</td><td>{number(p.y!)}</td><td>{p.count !== undefined ? `n=${p.count}` : ""}{p.error !== undefined ? ` ± ${number(p.error)} std` : ""}{p.value !== undefined ? `${chart.color_value}=${number(p.value)}` : ""}</td></tr>))}</tbody></table></div></details>}
  </article>;
}
