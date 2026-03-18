#!/usr/bin/env bash
set -euo pipefail
cd /home/saisamarth/projects/browser-agent
API_KEY_VALUE="$(grep '^GLM_API_KEY=' ~/.hermes/.env | head -1 | cut -d= -f2-)"
if [[ "$API_KEY_VALUE" == FILE:* ]]; then
  API_KEY_VALUE="$(cat "${API_KEY_VALUE#FILE:}")"
fi
TMP_CONFIG="$(mktemp /tmp/phase1-glm47-smoke-XXXX.yaml)"
.venv/bin/python - "$API_KEY_VALUE" "$TMP_CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
api_key = sys.argv[1]
out_path = Path(sys.argv[2])
base = yaml.safe_load(Path('configs/rollout_config_phase1_glm47.yaml').read_text())
base.setdefault('teacher_api', {})
base['teacher_api']['api_key'] = api_key
base['teacher_api'].pop('api_key_env_var', None)
out_path.write_text(yaml.safe_dump(base, sort_keys=False))
print(out_path)
PY
.venv/bin/python scripts/run_parallel_miniwob.py   --config-template "$TMP_CONFIG"   --task-list-file data/task_lists/phase1_smoke.txt   --workers 2   --episodes-per-task 1   --run-name-prefix miniwob_phase1_smoke_glm47
rm -f "$TMP_CONFIG"
