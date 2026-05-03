from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra

from Training.runtime import run_experiment
from utils.path_config import load_local_env


load_local_env()


@hydra.main(version_base=None, config_path="../../Configs", config_name="vanilla")
def main(config) -> None:
    run_experiment(config)


if __name__ == "__main__":
    main()
