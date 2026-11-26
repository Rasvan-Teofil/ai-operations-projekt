"""One audited retry after a real reboot; no reset, reboot or infinite restart loop."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'runtime/optimizer_ablation/resume-after-reboot.json'
UNIT = 'article-optimizer-ablation-resume.service'


def may_resume(state, boot_id, protocol_sha):
    return (state['state'] == 'armed' and state['failed_boot_id'] != boot_id
            and state['protocol_sha256'] == protocol_sha)


def main():
    state = json.loads(STATE.read_text())
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    protocol_sha = hashlib.sha256((ROOT / 'reports/optimizer_ablation/protocol-v1.json').read_bytes()).hexdigest()
    if not may_resume(state, boot_id, protocol_sha):
        print('No retry: requires an armed audit, unchanged protocol and a new boot.', flush=True)
        return
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    python = str(ROOT / '.venv/bin/python')
    for attempt in range(60):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and int(result.stdout.strip().splitlines()[0]) >= 12288:
                probe = subprocess.run([python, '-c', 'import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); x=torch.ones((16,16),device="cuda"); assert torch.isfinite(x@x).all(); torch.cuda.synchronize()'],
                                       cwd=ROOT, env=env, timeout=60)
                if probe.returncode == 0:
                    break
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        time.sleep(5)
    else:
        raise RuntimeError('GPU still unavailable after reboot; no training retry started')
    # Only this already audited attempt may restart. Further failures need a new audit.
    state.update(state='consumed', resumed_boot_id=boot_id)
    STATE.write_text(json.dumps(state, indent=2) + '\n')
    subprocess.run(['systemctl', '--user', 'disable', UNIT], check=True)
    os.chdir(ROOT)
    os.execve(python, [python, '-m', 'src.studies.optimizer_ablation', 'worker'], env)


if __name__ == '__main__':
    main()
