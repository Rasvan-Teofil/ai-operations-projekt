"""Bounded read-only GPU/PCIe telemetry; does not change clocks, power or drivers."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['temperature.gpu', 'power.draw', 'power.limit', 'utilization.gpu',
          'memory.used', 'memory.total', 'pstate', 'clocks.gr', 'clocks.mem',
          'pcie.link.gen.current', 'pcie.link.width.current']
UNITS = ['C', 'W', 'W', 'percent', 'MiB', 'MiB', 'state', 'MHz', 'MHz', 'generation', 'lanes']


def parse_gpu(stdout):
    rows = list(csv.reader(stdout.strip().splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != len(FIELDS):
        raise ValueError('Expected exactly one GPU and the declared CSV columns')
    return dict(zip(FIELDS, (v.strip() for v in rows[0])))


def read(path):
    try:
        return path.read_text().strip()
    except (OSError, UnicodeError):
        return None


def terminal_worker(state):
    return state in ('complete', 'failed', 'stopped')


def sample(output):
    event = dict(time_utc=datetime.now(timezone.utc).isoformat())
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(FIELDS), '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=8)
        event['nvidia_smi_returncode'] = result.returncode
        if result.returncode == 0:
            event['gpu'] = parse_gpu(result.stdout)
        else:
            event['gpu_error'] = (result.stdout + result.stderr).strip()[:2000]
    except (subprocess.TimeoutExpired, ValueError, OSError) as exc:
        event['gpu_error'] = str(exc)
    event['pcie'] = {}
    for device in ('0000:01:00.0', '0000:00:01.1'):
        folder = Path('/sys/bus/pci/devices') / device
        values = {name: read(folder / name) for name in ['current_link_speed', 'current_link_width', 'power/runtime_status']}
        values['aer'] = {p.name: read(p) for p in folder.glob('aer_*')}
        event['pcie'][device] = values
    event['cpu_temperature_millicelsius'] = {}
    for folder in Path('/sys/class/hwmon').glob('hwmon*'):
        if read(folder / 'name') == 'k10temp':
            event['cpu_temperature_millicelsius'] = {p.name: read(p) for p in folder.glob('temp*_input')}
    try:
        worker = json.loads((ROOT / 'runtime/optimizer_ablation/worker.json').read_text())
        event['worker'] = {k: worker.get(k) for k in ('state', 'current_run')}
    except (OSError, ValueError):
        event['worker'] = {'state': 'unknown'}
    (output / 'latest-health.json').write_text(json.dumps(event, indent=2) + '\n')
    return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=int, default=10800)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.duration < 1 or args.interval < 1:
        parser.error('duration and interval must be positive; interval at least one second')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stopped = False

    def stop(*unused):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (output / 'capture.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_log = (output / 'live-kernel.log').open('a')
        journal = subprocess.Popen(['journalctl', '-k', '-f', '-n', '0', '--no-pager', '-o', 'short-iso'],
                                   stdout=journal_log, stderr=journal_log)
        started = time.monotonic()
        count = 0
        try:
            with (output / 'health-samples.jsonl').open('a', buffering=1) as stream:
                while not stopped and time.monotonic() - started < args.duration:
                    before = time.monotonic()
                    event = sample(output)
                    stream.write(json.dumps(event, allow_nan=False) + '\n')
                    stream.flush(); os.fsync(stream.fileno())
                    count += 1
                    if args.once or terminal_worker(event['worker']['state']) or 'gpu_error' in event:
                        break
                    time.sleep(max(0, args.interval - (time.monotonic() - before)))
        finally:
            journal.terminate()
            try:
                journal.wait(timeout=5)
            except subprocess.TimeoutExpired:
                journal.kill(); journal.wait()
            journal_log.close()
            (output / 'capture-status.json').write_text(json.dumps(dict(
                samples=count, finished_at=datetime.now(timezone.utc).isoformat(),
                duration_seconds=time.monotonic() - started, stopped_by_signal=stopped), indent=2) + '\n')


if __name__ == '__main__':
    main()
