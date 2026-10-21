"""A field parameter that takes several names - a tuple of strings - is offered
to the designer as a list, not dropped."""

from bluesky_sandbox.ui.designer import catalog


def _params(name):
    field = next(f for f in catalog.action_fields() + catalog.obs_fields() if f["name"] == name)
    return {p["name"]: p for p in field["params"]}


def test_a_tuple_of_names_is_offered_as_a_list():
    params = _params("DirectToWaypoint")
    assert params["after"]["many"] is True
    assert params["after"]["default"] == []
    assert "many" not in params["waypoint"]


def test_a_single_value_is_not_a_list():
    assert all(not p.get("many") for p in _params("AltFt").values())
