"""Macro forecast agent: guarded LangGraph agent over oil and UK CPI forecasting tools."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor"
VENDOR_DATA = ROOT / "vendor_data"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
