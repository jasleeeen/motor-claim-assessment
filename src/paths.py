from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW_POLICIES = DATA / "raw_policies"    
SAMPLE_POLICIES = DATA / "sample_policies"   
TEST_CLAIMS = DATA / "test_claims.json"

OUTPUTS = ROOT / "outputs"    
WEIGHTS = ROOT / "weights"    