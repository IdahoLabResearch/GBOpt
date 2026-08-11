# Copyright 2025, Battelle Energy Alliance, LLC, ALL RIGHTS RESERVED

"""Shared default limits for exact crystallographic construction.

This private module centralizes user-facing default values that are shared across the
crystallography adapters, five-DOF exactification, ``GBMaker``, and the ``gb_params``
command-line interface.

The constants define defaults only. Validation and enforcement remain with the entry
point or construction layer that owns each invariant:

* primitive CSL area-index enforcement belongs to ``embedding.py``;
* exact P/Q determinant enforcement belongs to ``embedding.py``;
* adapter-level exception translation belongs to ``boundary.py``;
* command-line parsing belongs to ``gb_params.py``; and
* ``GBMaker`` argument validation belongs to ``GBMaker.py``.

The module is private and its constants are not part of
``GBOpt.crystallography.__init__``.
"""

from __future__ import annotations

DEFAULT_MAX_PRIMITIVE_AREA_INDEX: int = 10_000
DEFAULT_MAX_PQ_DETERMINANT: int = 10_000


__all__ = [
    "DEFAULT_MAX_PRIMITIVE_AREA_INDEX",
    "DEFAULT_MAX_PQ_DETERMINANT",
]
