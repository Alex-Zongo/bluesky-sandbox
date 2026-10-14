# Queryables

Queryables are named scenario entities—such as waypoints, routes, or airspace regions—referenced by observation fields. Fields declare the specific types and counts of entities they require, while the scenario supplies them by name. This abstraction allows identical field definitions to function across diverse task setups.


## `bluesky_sandbox.sim.queryables.base`

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.queryables.base
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.queryables.waypoints`

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.queryables.waypoints
   :members:
   :show-inheritance:
```

## `bluesky_sandbox.sim.arrival`

A waypoint is a place: how an aircraft crosses it - its altitude, speed and their windows, fly-over, a target arrival time - is its route step's, so two routes may cross one fix each their own way. A query's `.target` is this aircraft's `RouteFix` there, its `Crossing` as its route states it; an aircraft not routed through the waypoint gets the place alone. A route step's `arrival_slack_s` - seconds, or a distribution of them - gives each aircraft flying it a target arrival time there, assigned at spawn. `obs.ActiveRouteWaypointTimeToGoS`, `obs.ActiveRouteWaypointArrivalErrorS` and `obs.ActiveRouteWaypointHasArrivalTime` read it back, for the active fix or, with `route_offset`, a later one.

A step with `rta: True` has own navigation meet its time: it is handed to BlueSky's RTA, which VNAV turns into a speed - remaining distance over time left, with acceleration and tailwind - within the aircraft's envelope, early and late alike. A speed clearance takes the aircraft off it; resuming LNAV+VNAV hands it back. What speed cannot fix, two fields say in seconds: `obs.ActiveRouteWaypointTimeToAbsorbS`, the earliness left even at minimum speed (only a longer path loses it), and `obs.ActiveRouteWaypointUnrecoverableLateS`, the lateness left even at maximum speed.

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.arrival
   :members:
```

## `bluesky_sandbox.sim.queryables.regions`

```{eval-rst}
.. automodule:: bluesky_sandbox.sim.queryables.regions
   :members:
   :show-inheritance:
```
