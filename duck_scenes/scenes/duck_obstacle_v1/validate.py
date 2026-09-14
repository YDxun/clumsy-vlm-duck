#!/usr/bin/env python3
import sys
from pathlib import Path
HERE=Path(__file__).resolve()
sys.path.insert(0,str(HERE.parents[2]))
from validation_core import main
if __name__=="__main__":
    raise SystemExit(main(package_root=HERE.parents[2]))
