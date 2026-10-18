"""Adapters that convert existing evaluation logs into the policy-diff format.

  table    any CSV / TSV / JSONL with one row per episode (column mapping)
  lerobot  lerobot-eval eval_info.json, any LeRobot environment
  libero   LIBERO goal-condition logs, with lerobot-eval output and Conductor harness events

To log directly from an evaluation loop, use policy_diff.log.RunWriter.
"""

from .lerobot import import_lerobot
from .libero import import_libero
from .table import import_table

__all__ = ["import_lerobot", "import_libero", "import_table"]
