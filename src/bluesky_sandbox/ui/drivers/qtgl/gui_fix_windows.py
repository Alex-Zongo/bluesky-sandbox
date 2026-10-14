"""The QtGL client's drawing of each fix's constraints (see :mod:`.fix_windows`):
a symbol at the fix and its label beside it - name, altitude, speed - on the
radar, from what the sim sends.

Imported only in the QtGL client process (``client``), standalone - the
sandbox package is never imported there (see ``client``): it is a BlueSky GUI
render object, added to the radar when the first labels arrive - the GUI is
up by then - and redrawn with each that does.
"""

from __future__ import annotations

import numpy as np
from bluesky import settings
from bluesky.network.subscriber import subscriber
from bluesky.ui import palette
from bluesky.ui.qtgl import glhelpers as glh

try:  # as part of the sandbox package
    from .fix_windows import LINE_CHARS, LINES, TOPIC
except ImportError:  # in the client process, beside it (see client)
    from fix_windows import LINE_CHARS, LINES, TOPIC

#: The most fixes it draws.
MAX_FIXES = 1000

# Clear of BlueSky's waypoint blue and of an alert's hue.
palette.set_default_colours(fixwindow=(150, 230, 180))


class FixWindows(glh.RenderObject, layer=5):
    """Each constrained fix's symbol and label (see the module)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.nfixes = 0
        self.symbols = glh.VertexArrayObject(glh.gl.GL_LINE_LOOP)
        self.labelbuf = glh.GLBuffer()
        self.labels = glh.Text(settings.text_size, (LINE_CHARS, LINES))

    def create(self):
        size = settings.wpt_size
        self.labelbuf.create(MAX_FIXES * LINE_CHARS * LINES, usage=glh.GLBuffer.UsagePattern.StaticDraw)
        # A diamond, in screen units.
        diamond = np.array(
            [(0.0, 0.6 * size), (0.6 * size, 0.0), (0.0, -0.6 * size), (-0.6 * size, 0.0)], dtype=np.float32
        )
        self.symbols.create(vertex=diamond, color=palette.fixwindow)
        self.symbols.set_attribs(lat=MAX_FIXES * 4, lon=MAX_FIXES * 4, instance_divisor=1)
        self.labels.create(
            self.labelbuf, self.symbols.lat, self.symbols.lon, palette.fixwindow, (size, 0.5 * size), instanced=True
        )

    def show(self, data: dict) -> None:
        """Take the sim's labels (``lat``, ``lon``, ``label``) as the ones drawn."""
        n = min(len(data.get("label", ())), MAX_FIXES)
        self.nfixes = n
        if not n:
            return
        self.symbols.update(
            lat=np.array(data["lat"][:n], dtype=np.float32), lon=np.array(data["lon"][:n], dtype=np.float32)
        )
        self.labelbuf.update(np.array("".join(data["label"][:n]), dtype=np.bytes_))

    def draw(self):
        if not self.nfixes:
            return
        self.shaderset.set_vertex_scale_type(self.shaderset.VERTEX_IS_SCREEN)
        self.symbols.draw(n_instances=self.nfixes)
        self.labels.draw(n_instances=self.nfixes)


@subscriber(topic=TOPIC, raw=True)
def on_fix_windows(**data) -> None:
    """The sim's labels - a dict, which BlueSky passes as keywords - onto
    the radar, adding it there the first time."""
    if not FixWindows.is_instantiated():
        glh.addvisual("FIXWINDOWS")
    FixWindows.implinstance().show(data)
