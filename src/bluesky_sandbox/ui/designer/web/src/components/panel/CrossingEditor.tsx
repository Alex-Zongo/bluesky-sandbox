// How a route step crosses its fix: an altitude and a speed - each with a
// window about it - fly-over, and a target arrival time. A fix has none of its
// own: two routes may cross one fix each their own way.
import type { SpecDict } from "../../api";
import { NumInput, OptValueField } from "./ValueField";

// A {waypoint, ...crossing} step, or its bare name once it states nothing.
export function stepOrName(step: SpecDict): SpecDict | string {
  return Object.keys(step).length === 1 ? step.waypoint : step;
}

// A value with its window, as a chart writes it: "5000" at, "≥5000" at or
// above, "≤220" or less, "4000–6000" a band; a drawn value by its kind.
function windowed(v: any, below: any, above: any, unit: string): string | null {
  if (v == null) return null;
  if (typeof v !== "number") {
    if (v.type === "range") return `${v.low}–${v.high}${unit}`;
    if (v.type === "envelope") return `env${unit}`;
    if (v.type === "start") return "start";
    return `drawn${unit}`;
  }
  const lo = typeof below === "number" ? v - below : null;
  const hi = typeof above === "number" ? v + above : null;
  if (lo != null && hi != null) return lo === hi ? `${v}${unit}` : `${lo}–${hi}${unit}`;
  if (lo != null) return `≥${lo}${unit}`;
  if (hi != null) return `≤${hi}${unit}`;
  return `${v}${unit}`;
}

// A step's crossing in a line: "≥5000 ft · ≤220 kt · fly-over · RTA".
export function crossingSummary(step: SpecDict): string {
  return [
    windowed(step.alt_ft, step.alt_below_ft, step.alt_above_ft, " ft"),
    windowed(step.speed_kts, step.speed_below_kts, step.speed_above_kts, " kt"),
    step.flyover ? "fly-over" : null,
    step.arrival_slack_s != null ? (step.rta ? "RTA" : "timed") : null,
  ]
    .filter(Boolean)
    .join(" · ");
}

// One side of a step's window: how far below or above its stated value the
// aircraft may be - 0 for "at or above / at or below", unset for no limit that
// side. It may be sampled, as the value may.
function WindowSide({
  step,
  field,
  label,
  hint,
  inc,
  onChange,
}: {
  step: SpecDict;
  field: string;
  label: string;
  hint: string;
  inc: number;
  onChange: (step: SpecDict) => void;
}) {
  return (
    <OptValueField
      label={label}
      hint={hint}
      step={inc}
      value={step[field]}
      defaultValue={0}
      onChange={(v) => {
        const next = { ...step };
        if (v == null) delete next[field];
        else next[field] = v;
        onChange(next);
      }}
    />
  );
}

function Flag({
  step,
  field,
  label,
  onChange,
  onClear,
}: {
  step: SpecDict;
  field: string;
  label: string;
  onChange: (step: SpecDict) => void;
  onClear?: (next: SpecDict) => void;
}) {
  return (
    <label className="radio">
      <input
        type="checkbox"
        checked={step[field] === true}
        onChange={(e) => {
          const next = { ...step };
          if (e.target.checked) next[field] = true;
          else {
            delete next[field];
            onClear?.(next);
          }
          onChange(next);
        }}
      />
      {label}
    </label>
  );
}

export function CrossingEditor({ step, onChange }: { step: SpecDict; onChange: (step: SpecDict) => void }) {
  const envelopeAlt = typeof step.alt_ft === "object" && step.alt_ft?.type === "envelope";
  return (
    <div>
      <OptValueField
        label="alt ft"
        step={500}
        allowEnvelope
        allowStart
        altLevels
        value={step.alt_ft}
        defaultValue={3000}
        onChange={(v) => {
          const next = { ...step };
          if (v == null) {
            // Its window goes with it; a speed stays (it needs no altitude).
            delete next.alt_ft;
            delete next.alt_below_ft;
            delete next.alt_above_ft;
          } else next.alt_ft = v;
          // reachable-from-spawn only applies to an envelope-sampled altitude.
          if (v == null || typeof v !== "object" || v.type !== "envelope") {
            delete next.reachable_from_spawn;
            delete next.reachable_vs_fraction;
            delete next.envelope_alt_floor_ft;
          }
          onChange(next);
        }}
      />
      {envelopeAlt && (
        <>
          <OptValueField
            label="envelope floor ft"
            hint="The lowest altitude the envelope draw may give; unset, 1000 ft"
            step={500}
            value={step.envelope_alt_floor_ft}
            defaultValue={2000}
            onChange={(v) => {
              const next = { ...step };
              if (v == null) delete next.envelope_alt_floor_ft;
              else next.envelope_alt_floor_ft = v;
              onChange(next);
            }}
          />
          <Flag
            step={step}
            field="reachable_from_spawn"
            label="reachable from spawn"
            onChange={onChange}
            onClear={(next) => delete next.reachable_vs_fraction}
          />
          <div className="muted small">
            Bounds the per-aircraft envelope altitude to what the aircraft can climb/descend to before
            reaching the fix (from its max vertical rate and distance to go), so every episode is
            altitude-feasible.
          </div>
          {step.reachable_from_spawn === true && (
            <>
              <div className="value-field">
                <div className="vf-head">
                  <span className="vf-label">vs fraction</span>
                  <span className="vf-spacer" />
                  <NumInput
                    className="vf-input"
                    step={0.05}
                    value={typeof step.reachable_vs_fraction === "number" ? step.reachable_vs_fraction : 1}
                    onChange={(v) => {
                      const next = { ...step };
                      // 1.0 is the runtime default -> omit; store only a tighter
                      // margin (validation requires 0 < f <= 1).
                      if (Number.isFinite(v) && v > 0 && v < 1) next.reachable_vs_fraction = v;
                      else delete next.reachable_vs_fraction;
                      onChange(next);
                    }}
                  />
                </div>
              </div>
              <div className="muted small">
                Fraction of max climb/descent rate to assume (0–1, default 1). Lower reserves margin for
                the turn-to-fix and speed matching.
              </div>
            </>
          )}
        </>
      )}
      {step.alt_ft != null && (
        <>
          <WindowSide step={step} field="alt_below_ft" label="alt below ft" inc={100} onChange={onChange}
            hint="How far below alt ft the aircraft may be there: 0 is 'at or above'; unset, no limit" />
          <WindowSide step={step} field="alt_above_ft" label="alt above ft" inc={100} onChange={onChange}
            hint="How far above alt ft the aircraft may be there: 0 is 'at or below'; unset, no limit" />
        </>
      )}
      <OptValueField
        label="speed kt"
        step={5}
        allowEnvelope
        value={step.speed_kts}
        defaultValue={220}
        onChange={(v) => {
          const next = { ...step };
          if (v == null) {
            delete next.speed_kts;
            delete next.speed_below_kts;
            delete next.speed_above_kts;
          } else next.speed_kts = v;
          onChange(next);
        }}
      />
      {step.speed_kts != null && (
        <>
          <WindowSide step={step} field="speed_below_kts" label="speed below kt" inc={5} onChange={onChange}
            hint="How far below speed kt (CAS) the aircraft may be there: 0 is 'N KT or more'; unset, no limit. Above the CAS/Mach crossover it is checked in Mach, converted at altitude" />
          <WindowSide step={step} field="speed_above_kts" label="speed above kt" inc={5} onChange={onChange}
            hint="How far above speed kt (CAS) the aircraft may be there: 0 is 'N KT or less'; unset, no limit. Above the CAS/Mach crossover it is checked in Mach, converted at altitude" />
        </>
      )}
      <Flag step={step} field="flyover" label="fly over" onChange={onChange} />
      <div className="muted small">
        Crosses the fix before it turns, rather than turning early to join the next leg.
      </div>
      <OptValueField
        label="arrival slack s"
        hint="A target arrival time over this fix on this route"
        step={10}
        value={step.arrival_slack_s}
        defaultValue={{ type: "range", low: 0, high: 120 }}
        onChange={(v) => {
          const next = { ...step };
          if (v == null) {
            delete next.arrival_slack_s;
            delete next.rta;
          } else next.arrival_slack_s = v;
          onChange(next);
        }}
      />
      {step.arrival_slack_s != null && (
        <>
          <div className="muted small">
            Each aircraft is due over the fix at its nominal time from where it spawns - flown at its spawn
            speed, or its climb or descent at full rate if longer - plus this slack, drawn per aircraft.
            Never sooner than it can fly flat out. Observed through the ActiveRouteWaypoint arrival fields.
          </div>
          <Flag step={step} field="rta" label="hand BlueSky the RTA" onChange={onChange} />
          <div className="muted small">
            BlueSky's FMS then flies to make the time; unset, it is a target for the agent alone.
          </div>
        </>
      )}
    </div>
  );
}
