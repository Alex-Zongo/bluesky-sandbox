// Editor for a single queryable (goal/restricted region or waypoint). The body
// is exposed as `QueryableBody` (used by the geometry inspector) grouped into
// Shape/Position · Reach · Appearance · Advanced sub-groups; `QueryableCard`
// wraps that body in a collapsible card for any legacy list view.
import type { SpecDict } from "../../api";
import ShapeEditor, { NumField } from "../ShapeEditor";
import { CollapsibleCard, EyeToggle } from "./Section";
import { FieldGroup } from "./FieldGroup";
import { Picker } from "./Picker";
import { ColorPicker } from "./ColorPicker";
import { OptValueField } from "./ValueField";

type QueryableBodyProps = {
  name: string;
  q: SpecDict;
  regionNames: string[];
  namedRegions: Record<string, SpecDict>;
  namedRegionNames: string[];
  // A new bounds for it: an area for a region, a point for a waypoint.
  onNewShape: () => void;
  resolveRegion: (name: string) => SpecDict | undefined;
  onEditRegion: (name: string, bounds: SpecDict) => void;
  shapeRefCount: (name: string) => number;
  onChange: (q: SpecDict) => void;
  onFocus: () => void;
};

// The grouped field editor for one queryable (no card chrome). The host supplies
// the name/eye/delete header.
export function QueryableBody({
  name,
  q,
  regionNames,
  namedRegions,
  namedRegionNames,
  onNewShape,
  resolveRegion,
  onEditRegion,
  shapeRefCount,
  onChange,
  onFocus,
}: QueryableBodyProps) {
  const appearance = (
    <FieldGroup title="Appearance">
      <label className="numfield inline">
        <span>color</span>
        <ColorPicker value={q.color} onChange={(color) => onChange({ ...q, color })} />
      </label>
      <VisualizationToggles
        shape={q.render_shape !== false}
        label={q.render_label !== false}
        onShape={(render_shape) => onChange({ ...q, render_shape })}
        onLabel={(render_label) => onChange({ ...q, render_label })}
      />
      {q.type === "waypoint" && (
        <WaypointTsas q={q} regionNames={regionNames} onChange={onChange} />
      )}
    </FieldGroup>
  );
  const advanced = (
    <FieldGroup title="Advanced">
      <label className="radio">
        <input
          type="checkbox"
          checked={q.track_temporal_state === true}
          onChange={(e) => onChange({ ...q, track_temporal_state: e.target.checked })}
        />
        track temporal state
      </label>
      <div className="muted small">
        Enables per-substep tracking for this queryable when fields use during-step, time, or step-minimum values.
      </div>
    </FieldGroup>
  );

  if (q.type === "waypoint") {
    return (
      <div className="queryable-body">
        <FieldGroup title="Position" defaultOpen>
          <WaypointPosition
            q={q}
            namedRegions={namedRegions}
            namedRegionNames={namedRegionNames}
            onNewPoint={onNewShape}
            resolveRegion={resolveRegion}
            onEditRegion={onEditRegion}
            shapeRefCount={shapeRefCount}
            onChange={onChange}
            onFocus={onFocus}
          />
        </FieldGroup>
        <FieldGroup title="Reach">
          <OptValueField
            label="reach radius nm"
            hint="How near an aircraft must pass to have reached it"
            step={0.1}
            value={q.reach_radius_nm}
            defaultValue={1}
            onChange={(v) => onChange({ ...q, reach_radius_nm: v })}
          />
          <div className="muted small">
            How each route crosses it - altitude, speed, fly-over, arrival time - is set on the route's step.
          </div>
        </FieldGroup>
        {appearance}
        {advanced}
      </div>
    );
  }
  return (
    <div className="queryable-body">
      <FieldGroup title="Shape" defaultOpen>
        <ShapeEditor
          shape={q.shape}
          onChange={(b) => onChange({ ...q, shape: b })}
          onFocus={onFocus}
          regionNames={namedRegionNames}
          areaOnly
          requireRef
          onNewRegion={onNewShape}
          resolveRegion={resolveRegion}
          onEditRegion={onEditRegion}
          refCount={q.shape?.ref ? shapeRefCount(q.shape.ref) : undefined}
        />
      </FieldGroup>
      {appearance}
      {advanced}
    </div>
  );
}

export function QueryableCard({
  hidden,
  onToggleHidden,
  onRename,
  selected,
  onSelectCard,
  onRemove,
  ...body
}: QueryableBodyProps & {
  hidden: boolean;
  onToggleHidden: () => void;
  onRename: (n: string) => void;
  selected?: boolean;
  onSelectCard?: () => void;
  onRemove: () => void;
}) {
  return (
    <CollapsibleCard
      selected={selected}
      onSelect={onSelectCard}
      header={<>
        <input className="name-input" defaultValue={body.name} onBlur={(e) => onRename(e.target.value)} />
        <span className="kind-tag">{body.q.type === "waypoint" ? "waypoint" : "region"}</span>
        <EyeToggle hidden={hidden} onToggle={onToggleHidden} />
        <button className="link danger" onClick={onRemove}>
          ✕
        </button>
      </>}
    >
      <QueryableBody {...body} />
    </CollapsibleCard>
  );
}

export function VisualizationToggles({
  shape,
  label,
  onShape,
  onLabel,
}: {
  shape: boolean;
  label: boolean;
  onShape: (v: boolean) => void;
  onLabel: (v: boolean) => void;
}) {
  return (
    <div className="row">
      <label className="radio">
        <input type="checkbox" checked={shape} onChange={(e) => onShape(e.target.checked)} />
        show shape
      </label>
      <label className="radio">
        <input type="checkbox" checked={label} disabled={!shape} onChange={(e) => onLabel(e.target.checked)} />
        show label
      </label>
    </div>
  );
}

// Where a waypoint is: its point bounds - at a lat/lon or a navdb fix, placed
// anew each episode, in a group - edited here in place. A placed point can be
// drawn once an episode for every aircraft, or for each aircraft on its own.
function WaypointPosition({
  q,
  namedRegions,
  namedRegionNames,
  onNewPoint,
  resolveRegion,
  onEditRegion,
  shapeRefCount,
  onChange,
  onFocus,
}: {
  q: SpecDict;
  namedRegions: Record<string, SpecDict>;
  namedRegionNames: string[];
  onNewPoint: () => void;
  resolveRegion: (name: string) => SpecDict | undefined;
  onEditRegion: (name: string, bounds: SpecDict) => void;
  shapeRefCount: (name: string) => number;
  onChange: (q: SpecDict) => void;
  onFocus: () => void;
}) {
  const ref = q.shape?.ref;
  const placed = !!(ref && namedRegions[ref]?.placement);
  return (
    <div>
      <ShapeEditor
        shape={q.shape ?? {}}
        onChange={(b) => onChange({ ...q, shape: b })}
        onFocus={onFocus}
        regionNames={namedRegionNames}
        pointOnly
        requireRef
        onNewRegion={onNewPoint}
        resolveRegion={resolveRegion}
        onEditRegion={onEditRegion}
        refCount={ref ? shapeRefCount(ref) : undefined}
      />
      {placed && (
        <div className="sample-region">
          <label className="numfield inline">
            <span>drawn</span>
            <Picker
              searchable={false}
              placeholder="once an episode"
              value={q.sample_per === "aircraft" ? "aircraft" : "episode"}
              onChange={(v) => {
                const next = { ...q };
                if (v === "aircraft") next.sample_per = "aircraft";
                else delete next.sample_per;
                onChange(next);
              }}
              options={[
                { value: "episode", label: "once an episode", description: "where the point is placed, for every aircraft" },
                { value: "aircraft", label: "for each aircraft", description: "each aircraft its own spot in the point's region" },
              ]}
            />
          </label>
          <div className="muted small">
            {q.sample_per === "aircraft"
              ? "when this waypoint is on a spawn route, each aircraft draws its own target in the point's region at spawn; query fields read that target from BlueSky."
              : "the point's spot this episode, the same for every aircraft and whatever else uses the point; step the map's episode to see another."}
          </div>
        </div>
      )}
    </div>
  );
}

// The TSAS strip toggle + which region's aircraft it tracks.
function WaypointTsas({
  q,
  regionNames,
  onChange,
}: {
  q: SpecDict;
  regionNames: string[];
  onChange: (q: SpecDict) => void;
}) {
  return (
    <>
      <label className="radio">
        <input
          type="checkbox"
          checked={q.render_tsas ?? q.render_shape !== false}
          onChange={(e) => onChange({ ...q, render_tsas: e.target.checked })}
        />
        show TSAS strip
      </label>
      {(q.render_tsas ?? q.render_shape !== false) && (
        <label className="numfield inline">
          <span>TSAS bound</span>
          <Picker
            placeholder="all aircraft"
            value={q.tsas_region ?? ""}
            onChange={(v) => onChange({ ...q, tsas_region: v || undefined })}
            options={[{ value: "", label: "all aircraft" }, ...regionNames.map((name) => ({ value: name }))]}
          />
        </label>
      )}
    </>
  );
}

// An optional numeric constraint: a checkbox toggles it on/off. Off -> the
// value becomes undefined (dropped from the spec), so it can be cleared again.
export function OptNumField({
  label,
  value,
  step,
  min,
  defaultValue,
  onChange,
}: {
  label: string;
  value: number | null | undefined;
  step?: number;
  min?: number;
  defaultValue: number;
  onChange: (v: number | undefined) => void;
}) {
  const on = value != null && Number.isFinite(value);
  return (
    <div className="opt-field">
      <label className="radio">
        <input type="checkbox" checked={on} onChange={(e) => onChange(e.target.checked ? defaultValue : undefined)} />
        {label}
      </label>
      {on && <NumField label="" value={value as number} step={step} min={min} onChange={(v) => onChange(v)} />}
    </div>
  );
}


