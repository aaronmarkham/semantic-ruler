"""Page parsing for the doc-ordering scripts. The code lives in semantic_ruler.ordering."""

import importlib.util
import sys
from pathlib import Path

# Load the module file directly: it needs only the standard library, and these
# scripts are also run with `python -I`, where the package may not be importable.
_spec = importlib.util.spec_from_file_location(
    "semantic_ruler_ordering", Path(__file__).resolve().parents[2] / "semantic_ruler" / "ordering.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

StreamParser, parse, policy_summary, render = _mod.StreamParser, _mod.parse, _mod.policy_summary, _mod.render
PROMPT, TYPES = _mod.PROMPT, _mod.TYPES
