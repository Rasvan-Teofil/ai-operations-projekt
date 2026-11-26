"""Summarize all retained GPU-loss incidents, matching the current physical GPU."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

XID = re.compile(r'NVRM: Xid \(PCI:([0-9a-f:.]+)\):\s*(\d+)')
GPU_IDENTITY = re.compile(r'NVRM: GPU at PCI:([0-9a-f:.]+):\s+(GPU-[0-9a-f-]+)')


def analyze(events, current_uuid):
    counts, drivers, identities, losses = Counter(), {}, defaultdict(set), []
    kernels = {}
    boots, scanned = set(), 0
    for event in events:
        scanned += 1
        boot, message = event.get('_BOOT_ID', 'unknown'), event.get('MESSAGE', '')
        boots.add(boot)
        if message.startswith('Linux version '):
            kernels[boot] = message.split()[2]
        if 'NVRM: loading' in message:
            drivers[boot] = message
        if 'NVRM: GPU at PCI' in message:
            match = GPU_IDENTITY.search(message)
            if match:
                identities[boot, match.group(1).split('.')[0]].add(match.group(2))
        match = XID.search(message)
        if match:
            address = match.group(1).split('.')[0]
            code = int(match.group(2)); counts[str(code)] += 1
            seconds = int(event['__REALTIME_TIMESTAMP']) / 1e6
            losses.append(dict(xid=code, pci_address=address, boot_id=boot, timestamp=seconds,
                               time_utc=datetime.fromtimestamp(seconds, timezone.utc).isoformat(), message=message))
    clusters, last_cluster = [], {}
    for loss in losses:
        key = (loss['boot_id'], loss['pci_address'])
        known = identities[key]
        loss['same_physical_gpu_uuid_confirmed'] = current_uuid in known if known else None
        loss['driver_banner'] = drivers.get(loss['boot_id'])
        loss['kernel'] = kernels.get(loss['boot_id'])
        if loss['xid'] != 79:
            continue
        previous = last_cluster.get(key)
        if previous is None or loss['timestamp'] - clusters[previous]['last_timestamp'] > 60:
            clusters.append(dict(boot_id=loss['boot_id'], pci_address=loss['pci_address'], first_time_utc=loss['time_utc'],
                                 last_timestamp=loss['timestamp'], xid79_messages=1,
                                 same_physical_gpu_uuid_confirmed=loss['same_physical_gpu_uuid_confirmed'],
                                 driver_banner=loss['driver_banner'], kernel=loss['kernel']))
            last_cluster[key] = len(clusters) - 1
        else:
            clusters[previous]['xid79_messages'] += 1
            clusters[previous]['last_timestamp'] = loss['timestamp']
    return dict(kernel_events_scanned=scanned, retained_boots=len(boots), xid_counts=dict(counts),
                xid79_clusters=clusters, events=losses,
                caveat='Retained logs only; missing UUID/banner is unknown. Multiple messages within 60 seconds in one boot count as one incident.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    gpu_uuid = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    if len(gpu_uuid.splitlines()) != 1:
        raise RuntimeError('Specify a single-GPU host for physical identity matching')
    # `journalctl -k` silently implies current boot; the transport match retains history.
    process = subprocess.Popen(['journalctl', '--no-pager', '-o', 'json', '_TRANSPORT=kernel'], stdout=subprocess.PIPE, text=True)
    result = analyze((json.loads(line) for line in process.stdout), gpu_uuid)
    if process.wait() != 0:
        raise RuntimeError('Incomplete journal export')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('events', 'xid79_clusters')}))


if __name__ == '__main__':
    main()
