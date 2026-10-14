// Route editing: the named-route library, route sampling (global), and the
// per-region route override control. Routes are ordered lists of waypoint
// queryable names. Hovering a library row highlights that route on the map; the
// eye toggle hides it from the map (view-only).
import { shares } from "../../shares";
import { useState } from "react";
import type { SpecDict } from "../../api";
import { NumField } from "../ShapeEditor";
import { CollapsibleCard, EyeToggle } from "./Section";
import { Picker } from "./Picker";
import { ColorPicker } from "./ColorPicker";
import { type NamedRoute, routeColor, routeSteps, withRouteColor, withRouteSteps } from "../../route/namedRoute";
import { type Choice, type RouteStep, type WpStep, isChoice, isSubroute, isWpStep, stepName } from "../../route/routeGraph";
import { CrossingEditor, crossingSummary, stepOrName } from "./CrossingEditor";

function waypointRouteFallback(waypointNames: string[]): string[] {
  return waypointNames.length ? [waypointNames[0]] : [];
}

function routeMode(route: any, allowInherit: boolean): "inherit" | "none" | "fixed" | "named" | "sample" {
  if (route == null) return allowInherit ? "inherit" : "none";
  if (Array.isArray(route)) return "fixed";
  if (route?.type === "categorical") return "sample";
  return "named";
}

function normalizeWeightedRoute(route: any, routeNames: string[]): SpecDict | null {
  const previous = route?.type === "categorical" ? route.weights ?? {} : {};
  const weights: Record<string, number> = {};
  for (const name of routeNames) {
    const value = Number(previous[name] ?? 1);
    weights[name] = value > 0 ? value : 1;
  }
  return routeNames.length ? { type: "categorical", weights } : null;
}

function renameRouteRefs(route: any, oldName: string, newName: string): any {
  if (route === oldName) return newName;
  if (route?.type === "categorical") {
    const weights = { ...(route.weights ?? {}) };
    if (oldName in weights) {
      weights[newName] = weights[oldName];
      delete weights[oldName];
    }
    return { ...route, weights };
  }
  // A fixed step list: rename {route: oldName} subroute steps, recursing into
  // branch options.
  if (Array.isArray(route)) {
    return route.map((s) => {
      if (s && typeof s === "object" && Array.isArray((s as any).choice)) {
        return { ...s, choice: (s as any).choice.map((b: any) => renameRouteRefs(b, oldName, newName)) };
      }
      return s && typeof s === "object" && (s as any).route === oldName ? { route: newName } : s;
    });
  }
  return route;
}

function removeRouteRef(route: any, name: string): any {
  if (route === name) return null;
  if (route?.type === "categorical") {
    const weights = { ...(route.weights ?? {}) };
    delete weights[name];
    return Object.keys(weights).length ? { ...route, weights } : null;
  }
  // A fixed step list: drop {route: name} subroute steps, recursing into branch
  // options.
  if (Array.isArray(route)) {
    return route
      .filter((s) => !(s && typeof s === "object" && (s as any).route === name))
      .map((s) =>
        s && typeof s === "object" && Array.isArray((s as any).choice)
          ? { ...s, choice: (s as any).choice.map((b: any) => removeRouteRef(b, name)) }
          : s,
      );
  }
  return route;
}

// A route step is a waypoint name (string), a {waypoint, ...crossing} step, a
// subroute reference {route: name} expanded inline, or a branch {choice: [option,
// ...]} where one option is taken (weighted; uniform if no weights). An option is
// itself an ordered step list, so branches diverge (SID transitions) or merge
// (STAR entries onto a shared trunk).
function RouteWaypointPicker({
  route,
  waypointNames,
  routeNames = [],
  selfName,
  onChange,
}: {
  route: RouteStep[];
  waypointNames: string[];
  routeNames?: string[];
  selfName?: string;
  onChange: (route: RouteStep[]) => void;
}) {
  const clean = route.filter((s) => {
    if (isChoice(s)) return true;
    if (isSubroute(s)) return routeNames.includes(s.route);
    if (isWpStep(s)) return waypointNames.includes(s.waypoint);
    return typeof s === "string" && waypointNames.includes(s);
  });
  const usedWaypoints = new Set(
    clean.filter((s) => typeof s === "string" || isWpStep(s)).map(stepName),
  );
  const usedSubs = new Set(clean.filter(isSubroute).map((s) => s.route));
  const addableWp = waypointNames.filter((n) => !usedWaypoints.has(n));
  // Subroutes exclude self (direct cycle) and ones already referenced.
  const addableSub = routeNames.filter((n) => n !== selfName && !usedSubs.has(n));
  const setStep = (index: number, next: RouteStep) =>
    onChange(clean.map((s, i) => (i === index ? next : s)));
  const add = (value: string) => {
    if (value === "branch:") onChange([...clean, { choice: [[], []] }]);
    else if (value.startsWith("rt:")) onChange([...clean, { route: value.slice(3) }]);
    else if (value.startsWith("wp:")) onChange([...clean, value.slice(3)]);
  };
  const remove = (index: number) => onChange(clean.filter((_, i) => i !== index));
  const move = (index: number, delta: number) => {
    const nextIndex = index + delta;
    if (nextIndex < 0 || nextIndex >= clean.length) return;
    const next = [...clean];
    [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
    onChange(next);
  };
  return (
    <div className="route-waypoints">
      <div className="route-chips">
        {clean.map((step, i) =>
          isChoice(step) ? (
            <BranchEditor
              key={i}
              step={step}
              seq={i + 1}
              canUp={i > 0}
              canDown={i < clean.length - 1}
              waypointNames={waypointNames}
              routeNames={routeNames}
              selfName={selfName}
              onMove={(d) => move(i, d)}
              onRemove={() => remove(i)}
              onChange={(next) => (next ? setStep(i, next) : remove(i))}
            />
          ) : isSubroute(step) ? (
            <span className="chip route-chip subroute-chip" key={i}>
              <span className="route-seq">{i + 1}</span>
              <span className="route-name">{`⤷ ${step.route}`}</span>
              <button className="chip-cfg" disabled={i === 0} title="move earlier" onClick={() => move(i, -1)}>↑</button>
              <button className="chip-cfg" disabled={i === clean.length - 1} title="move later" onClick={() => move(i, 1)}>↓</button>
              <button className="chip-x" title="remove" onClick={() => remove(i)}>✕</button>
            </span>
          ) : (
            <WaypointStepChip
              key={i}
              step={step as string | WpStep}
              seq={i + 1}
              canUp={i > 0}
              canDown={i < clean.length - 1}
              onMove={(d) => move(i, d)}
              onRemove={() => remove(i)}
              onChange={(next) => setStep(i, next)}
            />
          ),
        )}
        {clean.length === 0 && <span className="muted small">no steps yet</span>}
      </div>
      <Picker
        className="route-add"
        placeholder="+ add step…"
        onChange={add}
        options={[
          ...addableWp.map((name) => ({ value: `wp:${name}`, label: name, category: "waypoints" })),
          ...addableSub.map((name) => ({ value: `rt:${name}`, label: `⤷ ${name}`, category: "subroutes" })),
          { value: "branch:", label: "⎇ branch (choice)", category: "branch" },
        ]}
      />
    </div>
  );
}

// A waypoint step chip. The ⚙ toggle edits how this route crosses the fix; a
// step stating nothing reverts to the bare name so the spec stays minimal.
function WaypointStepChip({
  step,
  seq,
  canUp,
  canDown,
  onMove,
  onRemove,
  onChange,
}: {
  step: string | WpStep;
  seq: number;
  canUp: boolean;
  canDown: boolean;
  onMove: (delta: number) => void;
  onRemove: () => void;
  onChange: (next: string | WpStep) => void;
}) {
  const name = stepName(step);
  const over: WpStep = isWpStep(step) ? step : { waypoint: name };
  const [editing, setEditing] = useState(false);
  const badge = crossingSummary(over);
  return (
    <>
      <span className={`chip route-chip${badge ? " constrained-chip" : ""}`}>
        <span className="route-seq">{seq}</span>
        <span className="route-name">{name}</span>
        {badge && <span className="route-xing" title="how this route crosses it">{badge}</span>}
        <button
          className={`chip-cfg${editing ? " on" : ""}`}
          title="how this route crosses it"
          onClick={() => setEditing((v) => !v)}
        >
          ⚙
        </button>
        <button className="chip-cfg" disabled={!canUp} title="move earlier" onClick={() => onMove(-1)}>↑</button>
        <button className="chip-cfg" disabled={!canDown} title="move later" onClick={() => onMove(1)}>↓</button>
        <button className="chip-x" title="remove" onClick={onRemove}>✕</button>
      </span>
      {editing && (
        <div className="route-xing-edit">
          <span className="muted small">cross {name}</span>
          <CrossingEditor step={over} onChange={(next) => onChange(stepOrName(next) as string | WpStep)} />
        </div>
      )}
    </>
  );
}

// One {choice: [...]} branch step: each option is itself an editable step list
// (so options can nest subroutes / further branches), with a relative-likelihood
// weight. Options that share a junction waypoint diverge/merge there on the map.
function BranchEditor({
  step,
  seq,
  canUp,
  canDown,
  waypointNames,
  routeNames,
  selfName,
  onMove,
  onRemove,
  onChange,
}: {
  step: Choice;
  seq: number;
  canUp: boolean;
  canDown: boolean;
  waypointNames: string[];
  routeNames: string[];
  selfName?: string;
  onMove: (delta: number) => void;
  onRemove: () => void;
  onChange: (next: Choice | null) => void;
}) {
  const branches = step.choice;
  const weights = step.weights;
  const weightAt = (bi: number) => Number(weights?.[bi] ?? 1);
  const setBranch = (bi: number, steps: RouteStep[]) =>
    onChange({ ...step, choice: branches.map((b, j) => (j === bi ? steps : b)) });
  const addBranch = () =>
    onChange({ ...step, choice: [...branches, []], weights: weights ? [...weights, 1] : undefined });
  const removeBranch = (bi: number) => {
    if (branches.length <= 1) return onRemove();
    onChange({
      ...step,
      choice: branches.filter((_, j) => j !== bi),
      weights: weights ? weights.filter((_, j) => j !== bi) : undefined,
    });
  };
  const setWeight = (bi: number, v: number) => {
    const next = branches.map((_, j) => (j === bi ? Math.max(0, v) : weightAt(j)));
    // Keep the spec clean: equal weights mean uniform, so store none.
    const uniform = next.every((x) => x === next[0]);
    onChange({ ...step, weights: uniform ? undefined : next });
  };
  return (
    <div className="route-branch">
      <div className="route-branch-head">
        <span className="route-seq">{seq}</span>
        <span className="route-branch-title">⎇ branch</span>
        <span className="spacer" />
        <button className="chip-cfg" disabled={!canUp} title="move earlier" onClick={() => onMove(-1)}>↑</button>
        <button className="chip-cfg" disabled={!canDown} title="move later" onClick={() => onMove(1)}>↓</button>
        <button className="chip-x" title="remove branch" onClick={onRemove}>✕</button>
      </div>
      {branches.map((b, bi) => (
        <div className="route-branch-option" key={bi}>
          <div className="route-branch-option-head">
            <span className="route-branch-label">option {bi + 1}</span>
            <label className="branch-weight type-share" title="its weight: shares are each weight over the sum, so 3, 1 is 75 %, 25 %">
              <input
                type="number"
                min={0}
                step={0.25}
                key={`w-${bi}-${weightAt(bi)}`}
                defaultValue={weightAt(bi)}
                onBlur={(e) => {
                  const v = Number(e.target.value);
                  if (Number.isFinite(v)) setWeight(bi, v);
                }}
              />
              <span className="type-share-pct">{shares(branches.map((_, j) => weightAt(j)))[bi]} %</span>
            </label>
            <button className="chip-x" title="remove option" onClick={() => removeBranch(bi)}>✕</button>
          </div>
          <RouteWaypointPicker
            route={b}
            waypointNames={waypointNames}
            routeNames={routeNames}
            selfName={selfName}
            onChange={(steps) => setBranch(bi, steps)}
          />
        </div>
      ))}
      <button className="branch-add-option" onClick={addBranch}>+ option</button>
    </div>
  );
}

// At-a-glance assignment of named routes to spawn regions: rows are regions,
// columns are inherit / each named route. Selecting a cell sets that region's
// route directly. Fixed-waypoint ordering and sample weights stay editable on
// each region card (a region in one of those modes shows no column checked).
function RouteAssignmentTable({
  spawn,
  routeNames,
  onChange,
}: {
  spawn: SpecDict;
  routeNames: string[];
  onChange: (spawn: SpecDict) => void;
}) {
  const regions: SpecDict[] = spawn.regions ?? [];
  if (regions.length === 0 || routeNames.length === 0) return null;
  const setRegionRoute = (i: number, route: any) =>
    onChange({ ...spawn, regions: regions.map((r, j) => (j === i ? { ...r, route } : r)) });
  return (
    <div className="route-assign">
      <div className="sub-label">route assignment</div>
      <div className="muted small">pick a named route per spawn region; fixed/sample routes stay on each region card.</div>
      <div className="route-assign-list">
        {regions.map((r, i) => {
          const name = r.name ?? `region ${i + 1}`;
          const mode = routeMode(r.route, true);
          return (
            <label className="numfield inline" key={i}>
              <span className="route-assign-name" title={name}>{name}</span>
              {mode === "fixed" || mode === "sample" ? (
                <span className="muted small">{mode} route — on its card</span>
              ) : (
                <Picker
                  searchable={routeNames.length > 6}
                  placeholder="inherit global"
                  value={mode === "named" && typeof r.route === "string" ? r.route : ""}
                  onChange={(v) => setRegionRoute(i, v || null)}
                  options={[
                    { value: "", label: "inherit global" },
                    ...routeNames.map((rn) => ({ value: rn })),
                  ]}
                />
              )}
            </label>
          );
        })}
      </div>
    </div>
  );
}

export function RouteSpecControl({
  label,
  route,
  routeNames,
  waypointNames,
  allowInherit = false,
  onChange,
}: {
  label: string;
  route: any;
  routeNames: string[];
  waypointNames: string[];
  allowInherit?: boolean;
  onChange: (route: any) => void;
}) {
  const mode = routeMode(route, allowInherit);
  const setMode = (nextMode: string) => {
    if (nextMode === "inherit" || nextMode === "none") onChange(null);
    else if (nextMode === "fixed") onChange(Array.isArray(route) ? route : waypointRouteFallback(waypointNames));
    else if (nextMode === "named") onChange(typeof route === "string" ? route : (routeNames[0] ?? ""));
    else if (nextMode === "sample") onChange(normalizeWeightedRoute(route, routeNames));
  };
  return (
    <div className="route-control">
      <label className="numfield inline">
        <span>{label}</span>
        <Picker
          searchable={false}
          placeholder="mode"
          value={mode}
          onChange={setMode}
          options={[
            allowInherit
              ? { value: "inherit", label: "inherit global" }
              : { value: "none", label: "none" },
            { value: "fixed", label: "fixed waypoints" },
            { value: "named", label: "named route", disabled: !routeNames.length },
            { value: "sample", label: "sample named routes", disabled: !routeNames.length },
          ]}
        />
      </label>
      {mode === "fixed" && (
        <RouteWaypointPicker
          route={Array.isArray(route) ? route : []}
          waypointNames={waypointNames}
          routeNames={routeNames}
          onChange={onChange}
        />
      )}
      {mode === "named" && (
        <label className="numfield inline">
          <span>route</span>
          <Picker
            placeholder="choose route"
            value={typeof route === "string" ? route : ""}
            onChange={(v) => onChange(v || null)}
            options={routeNames.map((name) => ({ value: name }))}
          />
        </label>
      )}
      {mode === "sample" && (
        <div className="route-weights">
          {routeNames.map((name, i) => (
            <span className="route-weight-row" key={name}>
              <NumField
                label={name}
                min={0.01}
                step={0.25}
                value={Number(route?.weights?.[name] ?? 1)}
                onChange={(v) => {
                  const weights = { ...(route?.weights ?? {}) };
                  weights[name] = Math.max(0.01, v);
                  onChange({ type: "categorical", weights });
                }}
              />
              <span className="type-share-pct" title="its share of this region's aircraft">
                {shares(routeNames.map((n) => Number(route?.weights?.[n] ?? 1)))[i]} %
              </span>
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export function RouteSettings({
  spawn,
  waypointNames,
  hiddenElements,
  onToggleHidden,
  onHighlightRoute,
  filter = "",
  onFilter,
  onChange,
}: {
  spawn: SpecDict;
  waypointNames: string[];
  hiddenElements: Set<string>;
  onToggleHidden: (key: string) => void;
  onHighlightRoute: (name: string | null) => void;
  filter?: string;
  onFilter?: (s: string) => void;
  onChange: (spawn: SpecDict) => void;
}) {
  const routes: Record<string, NamedRoute> = spawn.routes ?? {};
  const routeNames = Object.keys(routes);
  const setRoutes = (nextRoutes: Record<string, NamedRoute>) => onChange({ ...spawn, routes: nextRoutes });
  const renameRoute = (oldName: string, newNameRaw: string) => {
    const newName = newNameRaw.trim();
    if (!newName || newName === oldName || routes[newName]) return;
    const nextRoutes: Record<string, NamedRoute> = {};
    // Rename the key, and rewrite {route: oldName} subroute steps in every route.
    for (const [name, route] of Object.entries(routes)) {
      nextRoutes[name === oldName ? newName : name] = withRouteSteps(route, renameRouteRefs(routeSteps(route), oldName, newName));
    }
    onChange({
      ...spawn,
      route: renameRouteRefs(spawn.route, oldName, newName),
      routes: nextRoutes,
      regions: (spawn.regions ?? []).map((r: SpecDict) => ({ ...r, route: renameRouteRefs(r.route, oldName, newName) })),
    });
  };
  const removeRoute = (name: string) => {
    const nextRoutes: Record<string, NamedRoute> = {};
    // Drop the route, and any {route: name} subroute steps that referenced it.
    for (const [key, route] of Object.entries(routes)) {
      if (key !== name) nextRoutes[key] = withRouteSteps(route, removeRouteRef(routeSteps(route), name));
    }
    onChange({
      ...spawn,
      route: removeRouteRef(spawn.route, name),
      routes: nextRoutes,
      regions: (spawn.regions ?? []).map((r: SpecDict) => ({ ...r, route: removeRouteRef(r.route, name) })),
    });
  };
  return (
    <div className="route-settings">
      <div className="sub-label">route library</div>
      {routeNames.length > 3 && onFilter && (
        <input className="sec-search" placeholder="filter routes…" value={filter} onChange={(e) => onFilter(e.target.value)} />
      )}
      {routeNames
        .filter((name) => !filter || name.toLowerCase().includes(filter.toLowerCase()))
        .map((name) => (
        <div
          key={name}
          onMouseEnter={() => onHighlightRoute(name)}
          onMouseLeave={() => onHighlightRoute(null)}
        >
          <CollapsibleCard
            header={<>
              <input className="name-input" defaultValue={name} onBlur={(e) => renameRoute(name, e.target.value)} />
              <span className="spacer" />
              <EyeToggle hidden={hiddenElements.has(`route:${name}`)} onToggle={() => onToggleHidden(`route:${name}`)} />
              <button className="link danger" title="delete route" onClick={() => removeRoute(name)}>✕</button>
            </>}
          >
            <RouteWaypointPicker
              route={routeSteps(routes[name])}
              waypointNames={waypointNames}
              routeNames={routeNames}
              selfName={name}
              onChange={(route) => setRoutes({ ...routes, [name]: withRouteSteps(routes[name], route) })}
            />
            <label className="numfield inline">
              <span>color</span>
              <ColorPicker
                value={routeColor(routes[name])}
                onChange={(color) => setRoutes({ ...routes, [name]: withRouteColor(routes[name], color) })}
              />
            </label>
          </CollapsibleCard>
        </div>
      ))}
      {routeNames.length === 0 && <div className="muted small">no named routes</div>}
      <button
        onClick={() => {
          let i = routeNames.length + 1;
          let name = `route_${i}`;
          while (routes[name]) name = `route_${++i}`;
          setRoutes({ ...routes, [name]: waypointRouteFallback(waypointNames) });
        }}
      >
        + route
      </button>
      <RouteAssignmentTable
        spawn={spawn}
        routeNames={routeNames}
        onChange={onChange}
      />
      <div className="sub-label">route sampling</div>
      <RouteSpecControl
        label="global route"
        route={spawn.route}
        routeNames={routeNames}
        waypointNames={waypointNames}
        onChange={(route) => onChange({ ...spawn, route })}
      />
    </div>
  );
}
