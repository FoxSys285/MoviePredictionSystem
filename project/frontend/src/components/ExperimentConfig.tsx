import type { LabGroup } from "../lib/lab";

export function AlgorithmSelection({
  group,
  selected,
  onChange,
  disabled,
  reductionEnabled = false,
  onReductionChange,
}: {
  group: LabGroup;
  selected: string[];
  onChange: (value: string[]) => void;
  disabled: boolean;
  reductionEnabled?: boolean;
  onReductionChange?: (enabled: boolean) => void;
}) {
  return (
    <fieldset className="lab-algorithm-picker" disabled={disabled}>
      <legend>Thuật toán chạy</legend>
      <div>
        {group.algorithms.map((algorithm) => {
          const method = group.id === "rating" && algorithm.id === "linear" ? "PCA" : null;
          return <div className="lab-algorithm-option" key={algorithm.id}>
          <label className={selected.includes(algorithm.id) ? "selected" : ""}>
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
          {method && onReductionChange && <button type="button" role="switch"
            className="lab-algorithm-reduction-switch"
            aria-label={`Bật/tắt ${method} cho ${algorithm.name}`}
            aria-checked={reductionEnabled}
            disabled={disabled || !selected.includes(algorithm.id)}
            onClick={() => onReductionChange(!reductionEnabled)}>
            {method}: {reductionEnabled ? "Bật" : "Tắt"}
          </button>}
          </div>;
        })}
      </div>
    </fieldset>
  );
}
