"""Older names of renamed fields, kept working: a design's named shapes were
once its "bounds", an element's shape its ``bounds``."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T", bound=type)


def renamed(**old_to_new: str) -> Callable[[T], T]:
    """Class decorator (outside ``@dataclass``): for each ``old=new``, field
    ``new`` is still taken as keyword ``old`` and read as attribute ``old``."""

    def apply(cls: T) -> T:
        init = cls.__init__

        @functools.wraps(init)
        def __init__(self, *args, **kwargs):
            # The older name wins when both come: ``dataclasses.replace(x,
            # old=...)`` passes every field by its new name too.
            for old, new in old_to_new.items():
                if old in kwargs:
                    kwargs[new] = kwargs.pop(old)
            init(self, *args, **kwargs)

        cls.__init__ = __init__
        for old, new in old_to_new.items():
            setattr(
                cls,
                old,
                property(
                    lambda self, _new=new: getattr(self, _new),
                    # Setting it sets the new one - refused on a frozen class.
                    lambda self, value, _new=new: setattr(self, _new, value),
                    doc=f"``{new}``'s older name.",
                ),
            )
        return cls

    return apply


def split(**old_to_pair: tuple[str, str]) -> Callable[[T], T]:
    """Class decorator (outside ``@dataclass``): for each ``old=(low, high)``,
    a field replaced by two - a symmetric tolerance by a window's two sides -
    keyword ``old`` still sets both (each unless given by its own name), and
    attribute ``old`` reads it back where the two agree (else ``None``)."""

    def apply(cls: T) -> T:
        init = cls.__init__

        @functools.wraps(init)
        def __init__(self, *args, **kwargs):
            for old, (low, high) in old_to_pair.items():
                if old in kwargs:
                    value = kwargs.pop(old)
                    kwargs.setdefault(low, value)
                    kwargs.setdefault(high, value)
            init(self, *args, **kwargs)

        cls.__init__ = __init__
        for old, (low, high) in old_to_pair.items():
            setattr(
                cls,
                old,
                property(
                    lambda self, _low=low, _high=high: (
                        getattr(self, _low) if getattr(self, _low) == getattr(self, _high) else None
                    ),
                    doc=f"``{low}`` and ``{high}``'s older, symmetric name.",
                ),
            )
        return cls

    return apply


def dropped(**old_to_why: str) -> Callable[[T], T]:
    """Class decorator (outside ``@dataclass``): keyword ``old`` - a field no
    longer there - is still taken, and ignored with a warning saying why."""

    def apply(cls: T) -> T:
        init = cls.__init__

        @functools.wraps(init)
        def __init__(self, *args, **kwargs):
            for old, why in old_to_why.items():
                if kwargs.pop(old, None) is not None:
                    warnings.warn(f"{cls.__name__}({old}=...) is ignored: {why}", DeprecationWarning, stacklevel=2)
            init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return apply
