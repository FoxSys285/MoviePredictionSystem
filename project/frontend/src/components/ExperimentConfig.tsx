import type { LabGroup } from "../lib/lab";

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
