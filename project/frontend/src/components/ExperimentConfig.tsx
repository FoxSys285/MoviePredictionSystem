import type { LabGroup, LabPreset } from "../lib/lab";

export function AlgorithmSelection({
  group,
  selected,
  onChange,
  disabled,
}: {
  group: LabGroup;
  selected: string[];
  onChange: (value: string[]) => void;
  disabled: boolean;
}) {
  return (
    <fieldset className="lab-algorithm-picker" disabled={disabled}>
      <legend>Thuật toán chạy</legend>
      <div>
        {group.algorithms.map((algorithm) => (
          <label key={algorithm.id} className={selected.includes(algorithm.id) ? "selected" : ""}>
            <input
              type="checkbox"
              checked={selected.includes(algorithm.id)}
              onChange={(event) => onChange(
                event.target.checked
                  ? [...selected, algorithm.id]
                  : selected.filter((id) => id !== algorithm.id),
              )}
            />
            <span>{algorithm.name}</span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export function ExperimentConfiguration({
  group,
  preset,
  algorithms,
}: {
  group: LabGroup;
  preset?: LabPreset;
  algorithms: string[];
}) {
  return (
    <section className="lab-preset" aria-label="Cấu hình preset chỉ đọc">
      <div className="lab-preset-heading">
        <div>
          <span className="lab-kicker">PRESET V{preset?.version ?? 1}</span>
          <h3>{preset?.label ?? "Preset chuẩn"}</h3>
        </div>
        <span className="lab-badge">Tuning tắt</span>
      </div>
      <dl className="lab-preset-common">
        {(preset?.common ?? []).map((setting) => (
          <div key={setting.label}>
            <dt>{setting.label}</dt>
            <dd>{setting.value || "Không có dữ liệu"}</dd>
          </div>
        ))}
      </dl>
      {algorithms.length > 0 ? (
        <div className="lab-preset-algorithms">
          {algorithms.map((id) => {
            const algorithm = group.algorithms.find((candidate) => candidate.id === id);
            const settings = preset?.algorithms[id]?.settings ?? [];
            return (
              <article key={id}>
                <h4>{algorithm?.name ?? id}</h4>
                {settings.length > 0 ? (
                  <dl>
                    {settings.map((setting) => (
                      <div key={setting.label}>
                        <dt>{setting.label}</dt>
                        <dd>{setting.value || "Không có dữ liệu"}</dd>
                      </div>
                    ))}
                  </dl>
                ) : (
                  <p>Không có tham số bổ sung.</p>
                )}
              </article>
            );
          })}
        </div>
      ) : (
        <p className="lab-preset-empty">Chọn ít nhất một thuật toán để xem preset.</p>
      )}
    </section>
  );
}
