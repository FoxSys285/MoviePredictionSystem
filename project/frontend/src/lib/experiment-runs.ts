import type { LabGroup, LabJob } from "./lab";

export function runAlgorithmIds(run: LabJob): string[] {
  const requested = run.request?.algorithms;
  if (requested?.length) return [...new Set(requested)];
  return [...new Set((run.result?.rows ?? []).map((row) => String(row.base_algorithm ?? row.algorithm)).filter(Boolean))];
}

export function runAlgorithmTitle(run: LabJob, group: LabGroup, translate: (value: string) => string): string {
  const ids = runAlgorithmIds(run);
  const names = ids.length >= 3 ? `${ids.length} ${translate("thuật toán")}` : ids.map((id) => {
    const name = group.algorithms.find((algorithm) => algorithm.id === id)?.name ?? id.replaceAll("_", " ");
    return name.replace(/\s+(?:regression|regressor)$/i, "");
  }).join(", ");
  const reduction = run.request?.reduction_choice === "pca" ? "PCA" : "Không giảm chiều";
  return `${translate("Hồi quy")} - ${names || translate("thuật toán")} - ${translate(reduction)}`;
}
