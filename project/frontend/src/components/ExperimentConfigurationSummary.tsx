import type { LabGroup, LabPreset } from "../lib/lab";

export function ExperimentConfigurationSummary({ group, preset, algorithms }: {
  group: LabGroup;
  preset?: LabPreset;
  algorithms: string[];
}) {
  const split = preset?.common.find((setting) => setting.label === "Train / Validation / Test");
  return (
    <section className="lab-section lab-configuration-summary" aria-labelledby="lab-configuration-title" data-testid="lab-configuration-summary">
      <h2 id="lab-configuration-title">Cấu hình thực nghiệm</h2>
      <dl className="lab-configuration-rows">
        <div>
          <dt>Train / Valid / Test</dt>
          <dd>{split?.value ?? "Đang tải cấu hình…"}</dd>
        </div>
        {algorithms.map((id) => {
          const algorithm = group.algorithms.find((item) => item.id === id);
          const config = preset?.algorithms[id];
          const settings = [...(config?.settings ?? [])];
          if (group.id === "rating" && id === "linear") {
            const reduction = config?.reduction;
            settings.push({label: "reduction", value: reduction
              ? `PCA(n_components=${reduction.components})`
              : "None"});
          }
          return (
            <div key={id}>
              <dt>{algorithm?.name ?? id}</dt>
              <dd>{settings.length
                ? settings.map((setting) => `${setting.label}=${setting.value}`).join(" | ")
                : preset ? "Tham số mặc định" : "Đang tải cấu hình…"}</dd>
            </div>
          );
        })}
      </dl>
    </section>
  );
}
