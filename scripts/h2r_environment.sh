#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
env_path=${1:?Usage: h2r_environment.sh ENV_PATH [PYTHON]}
python_bin=${2:-python3.11}
uv venv --python "$python_bin" --seed "$env_path"
# Legacy chumpy needs pip and NumPy while building; do not disable isolation globally.
uv pip install --python "$env_path/bin/python" numpy==1.26.4
uv pip sync --python "$env_path/bin/python" --no-build-isolation requirements-h2r.lock
MANIPTRANS_PROFILE=h2r uv pip install --python "$env_path/bin/python" --no-deps -e .
"$env_path/bin/python" -c 'from mjlab.sim import Simulation; from lib.rl.agent import PPOAgent; print("mjlab and native PPO imports OK")'
