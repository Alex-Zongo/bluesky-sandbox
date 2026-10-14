"""BlueSky's QtGL client, with the sandbox's own drawing loaded: what
``python -m bluesky --client <host>`` runs - BlueSky's own start, step by
step - with what the sandbox adds to the radar (``gui_fix_windows``) imported
once BlueSky is initialized, as its GUI modules expect.

Run as a script - ``python <this file> <host>`` - not as a module of the
sandbox package: importing the package loads BlueSky's stack commands before
``bs.init``, which then registers them again ("Attempt to reimplement ECHO
..."). Its drawing modules sit beside it and need nothing else of the
sandbox, so the script imports them as its own.
"""

from __future__ import annotations

import sys


def main() -> None:
    import bluesky as bs  # noqa: PLC0415
    from bluesky import cmdargs  # noqa: PLC0415

    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    sys.argv = [sys.argv[0], "--client", host]
    args = cmdargs.parse()
    bs.init(**args)

    from bluesky.ui import qtgl  # noqa: PLC0415

    import gui_fix_windows  # noqa: F401, PLC0415 - registers its drawing (beside this script)

    qtgl.start(hostname=args.get("hostname"))


if __name__ == "__main__":
    main()
