import { useId, useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import { usePreferences } from "../lib/preferences";

export function ExperimentDisclosure({ title, className, children }: {
  title: string; className: string; children: ReactNode;
}) {
  const { t } = usePreferences();
  const [expanded, setExpanded] = useState(true);
  const id = useId();
  return <section className={className} aria-labelledby={`${id}-title`}>
    <h3 id={`${id}-title`} className="experiment-disclosure-heading">
      <button type="button" aria-expanded={expanded} aria-controls={id}
        onClick={() => setExpanded((value) => !value)}>
        <span>{t(title)}</span>
        <span className="experiment-disclosure-action">{t(expanded ? "Thu gọn" : "Mở rộng")}
          <ChevronDown size={18} aria-hidden="true" />
        </span>
      </button>
    </h3>
    <div id={id} hidden={!expanded}>{children}</div>
  </section>;
}
