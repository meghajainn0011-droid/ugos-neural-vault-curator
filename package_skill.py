#!/usr/bin/env python3
"""
package_skill.py — Bundle the UGREEN AI NAS Skill for Devpost Round 1 submission.
"""
import zipfile
from pathlib import Path

SKILL_DIR = Path("/home/damon/job_hunter/active_competitions/ugreen_ai_skill")
OUTPUT_ZIP = SKILL_DIR / "ugos_neural_vault_curator_skill.zip"

FILES_TO_INCLUDE = [
    "SKILL.md",
    "proposal.md",
    "vault_curator.py",
    "test_vault_curator.py"
]

def create_package():
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in FILES_TO_INCLUDE:
            fpath = SKILL_DIR / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)
                print(f"Added {fname} ({fpath.stat().st_size} bytes)")
    print(f"\nPackage created: {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size} bytes)")

if __name__ == "__main__":
    create_package()
