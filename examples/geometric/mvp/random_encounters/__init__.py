"""Generated task package: Random Encounters.

Created by the bluesky-sandbox Environment Designer. ``scenario.py`` holds the
geometry/spawn/queryables; ``config.py`` holds static fields/settings;
``setup.py`` holds the module-level helpers; ``env.py`` holds env hooks.
"""

from .scenario import RandomEncountersScenario
from .env import RandomEncountersEnv

#: Design version, carried from the designer's metadata. Bump it there when
#: the design changes so a checkpoint or a run can be traced to what produced
#: it - the package name alone does not distinguish two edits of one design.
__version__ = "0"

Scenario = RandomEncountersScenario
Env = RandomEncountersEnv

__all__ = [
    "__version__",
    "RandomEncountersScenario",
    "RandomEncountersEnv",
    "Scenario",
    "Env",
]
