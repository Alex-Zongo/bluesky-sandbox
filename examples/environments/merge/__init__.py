"""Generated task package: merge.

Created by the bluesky-sandbox Environment Designer. ``scenario.py`` holds the
geometry/spawn/queryables; ``config.py`` holds static fields/settings;
``setup.py`` holds the module-level helpers; ``env.py`` holds env hooks.
"""

from .scenario import MergeScenario
from .env import MergeEnv

#: Design version, carried from the designer's metadata. Bump it there when
#: the design changes so a checkpoint or a run can be traced to what produced
#: it - the package name alone does not distinguish two edits of one design.
__version__ = "0"

Scenario = MergeScenario
Env = MergeEnv

__all__ = [
    "__version__",
    "MergeScenario",
    "MergeEnv",
    "Scenario",
    "Env",
]
