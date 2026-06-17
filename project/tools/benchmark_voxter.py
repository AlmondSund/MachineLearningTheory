#!/usr/bin/env python
from __future__ import annotations

import runpy
import sys
from pathlib import Path

impl = Path(__file__).with_name("_voxter_benchmark_impl.py")
sys.argv = [str(impl), *sys.argv[1:]]
runpy.run_path(str(impl), run_name="__main__")
