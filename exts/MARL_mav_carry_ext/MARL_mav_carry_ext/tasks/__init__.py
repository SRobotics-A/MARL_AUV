"""Package containing task implementations for various robotic environments."""

import os

##
# Register Gym environments.
##

# ManagerBased tasks create visualization markers at import time. In offline
# containers, those marker USDs may resolve to remote Omniverse URLs and fail
# before the requested task is even registered. Register DirectMARL tasks by
# default; opt into the legacy full import only when needed.
if os.environ.get("MARL_MAV_IMPORT_ALL_TASKS", "0") == "1":
    from isaaclab_tasks.utils import import_packages

    import_packages(__name__, ["utils"])
else:
    from . import directMARL  # noqa: F401
