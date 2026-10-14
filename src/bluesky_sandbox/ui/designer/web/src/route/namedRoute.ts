// A named route as the spec writes it: its steps, or - with a color - an
// object holding them ({type: "route", steps, color}); the library's Route.
// Everything that reads or edits a named route goes through these, so the
// color stays with the route whatever edits its steps.
import type { RouteStep } from "./routeGraph";

export type NamedRoute = RouteStep[] | { type: "route"; steps: RouteStep[]; color?: string };

export const routeSteps = (route: any): RouteStep[] =>
  Array.isArray(route) ? route : Array.isArray(route?.steps) ? route.steps : [];

export const routeColor = (route: any): string | undefined =>
  route && !Array.isArray(route) && typeof route.color === "string" ? route.color : undefined;

// The route with new steps, its color kept.
export const withRouteSteps = (route: any, steps: RouteStep[]): NamedRoute => {
  const color = routeColor(route);
  return color ? { type: "route", steps, color } : steps;
};

// The route in ``color`` - or, with none, the default colors' (its steps alone).
export const withRouteColor = (route: any, color: string | undefined): NamedRoute =>
  color ? { type: "route", steps: routeSteps(route), color } : routeSteps(route);

// Every step through each waypoint, across the design's routes - its named
// ones, the global one and each region's - however deep in a branch: what
// each route crosses it at. A bare name crosses it as stated nowhere.
export function waypointSteps(spawn: any): Record<string, Record<string, any>[]> {
  const out: Record<string, Record<string, any>[]> = {};
  const walk = (steps: any) => {
    for (const s of Array.isArray(steps) ? steps : []) {
      if (typeof s === "string") (out[s] ??= []).push({ waypoint: s });
      else if (typeof s?.waypoint === "string") (out[s.waypoint] ??= []).push(s);
      else if (Array.isArray(s?.choice)) s.choice.forEach((b: any) => walk(Array.isArray(b) ? b : [b]));
    }
  };
  for (const route of Object.values(spawn?.routes ?? {})) walk(routeSteps(route));
  walk(spawn?.route);
  for (const region of spawn?.regions ?? []) walk(region?.route);
  return out;
}
