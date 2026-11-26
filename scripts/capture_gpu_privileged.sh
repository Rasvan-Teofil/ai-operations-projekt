#!/bin/sh
# Read-only privileged evidence capture. No driver install, reset, firmware or PCI writes.
set -eu
if [ "$(id -u)" -ne 0 ]; then
    echo 'Run through the operating system administrator authentication (pkexec or sudo).' >&2
    exit 1
fi
requesting_uid=${PKEXEC_UID:-${SUDO_UID:-0}}
case "$requesting_uid" in ''|*[!0-9]*) exit 2 ;; esac
capture_dir=$(mktemp -d /var/tmp/article-gpu-diagnostics.XXXXXX)
chmod 700 "$capture_dir"
date --iso-8601=seconds > "$capture_dir/captured-at.txt"
uname -a > "$capture_dir/kernel.txt"
lspci -D -s 0000:01:00.0 -vv > "$capture_dir/gpu-pcie.txt"
lspci -D -s 0000:00:01.1 -vv > "$capture_dir/root-port-pcie.txt"
dmidecode --type 39 > "$capture_dir/power-supply-smbios.txt"
dmidecode --type 9 > "$capture_dir/slots-smbios.txt"
journalctl -k -b 0 --no-pager -o short-iso > "$capture_dir/current-boot-kernel.txt"
journalctl -k -b -1 --no-pager -o short-iso > "$capture_dir/previous-boot-kernel.txt"
# Safe mode skips GPU debug dump and potentially hanging graphics queries.
# Do not use --extra-system-data: no SysRq or kernel-control writes are needed.
set +e
timeout 50 /usr/bin/nvidia-bug-report.sh --safe-mode --output-file "$capture_dir/nvidia-bug-report.log" > "$capture_dir/collector-output.txt" 2>&1
collector_status=$?
set -e
printf '%s\n' "$collector_status" > "$capture_dir/collector-exit-status.txt"
if [ "$requesting_uid" -ne 0 ]; then
    chown -R "$requesting_uid:$(id -g "$requesting_uid")" "$capture_dir"
fi
printf 'GPU_DIAGNOSTIC_BUNDLE=%s\n' "$capture_dir"
