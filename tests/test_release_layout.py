from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_release_layout() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "validate_release.py")], check=True)

