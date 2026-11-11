import { usePreferences } from "../lib/preferences";
import { metricNumber, type LabChart } from "../lib/lab";

const colors = ["#de81b0", "#55c6c0", "#b1a1ff", "#e8be71", "#6caae5", "#a1d68e"];
const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

export function LabChartFigure({ chart }: { chart: LabChart }) {
  const { language } = usePreferences();
  if(chart.kind!=="bar")return <ContinuousChart chart={chart} language={language}/>;
  const categories = [...new Set(chart.series.flatMap(s => s.points.map(p => p.x)))];
  const values = chart.series.flatMap(s => s.points.map(p => p.y)).filter(finite);
  const width = Math.max(760, categories.length*132+100), height = 410;
  const left = 82, right = 24, top = 56, bottom = 112;
  const minimum = Math.min(0,...values), maximum = Math.max(0,...values);
  const span = maximum-minimum || 1;
  const minY = minimum < 0 ? minimum-span*.2 : 0, maxY = maximum+span*.25;
  const y = (v: number) => height-bottom-(v-minY)/(maxY-minY)*(height-top-bottom);
  const slot = (width-left-right)/Math.max(1,categories.length);
  const barWidth = Math.min(52,slot*.7/Math.max(1,chart.series.length));
  const number = (v: unknown) => metricNumber(v,language);
  return <article className="lab-chart" data-chart-kind="bar" data-metric={chart.metric_key}>
    <div className="lab-chart-heading"><span>{String(chart.index+1).padStart(2,"0")}</span><small>{chart.split}</small></div>
    <h3>{chart.title}</h3>
    <div className="lab-plot-scroll"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={chart.title} className="lab-plot" style={{ minWidth: width, maxWidth: "none" }}>
      <title>{chart.title} · Validation và Test</title>
      {[0,1,2,3,4].map(i => { const v = minY+(maxY-minY)*i/4; return <g key={i}><line className="lab-gridline" x1={left} x2={width-right} y1={y(v)} y2={y(v)} /><text x={left-9} y={y(v)+4} textAnchor="end">{Number(v.toPrecision(4)).toLocaleString(language === "en" ? "en-US" : "vi-VN")}</text></g>; })}
      <line className="lab-zero-line" x1={left} x2={width-right} y1={y(0)} y2={y(0)} />
      {categories.map((category,i) => { const x=left+(i+.5)*slot; const text=String(category); return <text key={text} x={x} y={height-bottom+23} textAnchor="end" transform={`rotate(-28 ${x} ${height-bottom+23})`}><title>{text}</title>{text.length>22 ? text.slice(0,20)+"…" : text}</text>; })}
      <text x={(width+left-right)/2} y={height-9} textAnchor="middle">{chart.x_label}</text>
      {chart.series.map((s,si) => <g key={s.label} data-series={s.label}>{s.points.map((p,pi) => {
        const ci=categories.indexOf(p.x), groupWidth=barWidth*chart.series.length+8*(chart.series.length-1);
        const x=left+(ci+.5)*slot-groupWidth/2+si*(barWidth+8), center=x+barWidth/2;
        const valid=finite(p.y), labelY=valid ? y(p.y!)+(p.y!<0 ? 18 : -10) : y(0)-10;
        return <g key={pi}>{valid && <rect x={x} y={Math.min(y(0),y(p.y!))} width={barWidth} height={Math.max(.8,Math.abs(y(p.y!)-y(0)))} fill={colors[si%colors.length]} rx={2}><title>{s.label} · {p.x}: {p.y}</title></rect>}
          <text className={valid ? "lab-bar-value" : "lab-bar-missing"} data-value={valid ? p.y! : "N/A"} x={center} y={labelY} textAnchor="middle" transform={`rotate(-38 ${center} ${labelY})`}><title>{s.label} · {p.x}: {valid ? String(p.y) : "Không xác định"}</title>{number(p.y)}</text>
        </g>;
      })}</g>)}
    </svg></div>
    <div className="lab-legend lab-static-legend">{chart.series.map((s,i) => <span key={s.label}><i style={{ background:colors[i%colors.length] }} />{s.label}</span>)}</div>
  </article>;
}

function ContinuousChart({chart,language}:{chart:LabChart;language:string}) {
  const points=chart.series.flatMap(s=>s.points).filter(p=>finite(p.x)&&finite(p.y));
  const xs=points.map(p=>Number(p.x)),ys=points.map(p=>Number(p.y));
  const minX=Math.min(...xs,0),maxX=Math.max(...xs,1),minY=Math.min(...ys,0),maxY=Math.max(...ys,1);
  const x=(v:number)=>70+(v-minX)/(maxX-minX||1)*620,y=(v:number)=>320-(v-minY)/(maxY-minY||1)*260;
  return <article className="lab-chart" data-chart-kind={chart.kind}>
    <div className="lab-chart-heading"><span>{String(chart.index+1).padStart(2,"0")}</span><small>{chart.split}</small></div><h3>{chart.title}</h3>
    <svg viewBox="0 0 760 390" role="img" aria-label={chart.title} className="lab-continuous-plot"><title>{chart.title}</title>
      {[0,1,2,3,4].map(i=><g key={i}><line className="lab-gridline" x1="70" x2="690" y1={60+i*65} y2={60+i*65}/><text x="60" y={64+i*65} textAnchor="end">{metricNumber(maxY-(maxY-minY)*i/4,language)}</text><text x={70+i*155} y="344" textAnchor="middle">{metricNumber(minX+(maxX-minX)*i/4,language)}</text></g>)}
      {chart.series.map((s,i)=>{const valid=s.points.filter(p=>finite(p.x)&&finite(p.y));return <g key={s.label} fill={colors[i%colors.length]}>
        {chart.kind==="line"&&<polyline fill="none" stroke={colors[i%colors.length]} strokeWidth="2.5" points={valid.map(p=>`${x(Number(p.x))},${y(Number(p.y))}`).join(" ")}/>}
        {valid.map((p,j)=><circle key={j} cx={x(Number(p.x))} cy={y(Number(p.y))} r={chart.kind==="scatter"?3:2}><title>{s.label} · {p.x}: {metricNumber(p.y,language)}</title></circle>)}
      </g>;})}<text x="380" y="381" textAnchor="middle">{chart.x_label}</text><text x="70" y="24">{chart.y_label}</text>
    </svg><div className="lab-legend lab-static-legend">{chart.series.map((s,i)=><span key={s.label}><i style={{background:colors[i%colors.length]}}/>{s.label}</span>)}</div>
  </article>;
}
