"""
Stage 11: Generate the final submission CSV.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.submit.build_submission import main

if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "v2_final"
    main(tag)
