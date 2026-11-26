import pytest
from scripts.capture_gpu_health import parse_gpu, terminal_worker


def test_csv_preserves_unknown_values_and_rejects_ambiguous_device_layout():
    row = '63, 140.5, 180.0, 99, 9400, 16311, P1, 2812, N/A, 4, 4'
    parsed = parse_gpu(row)
    assert parsed['clocks.mem'] == 'N/A'
    assert parsed['pcie.link.width.current'] == '4'
    with pytest.raises(ValueError):
        parse_gpu(row + '\n' + row)
    with pytest.raises(ValueError):
        parse_gpu('GPU has fallen off the bus')


def test_capture_stops_when_training_ends_or_fails():
    assert terminal_worker('failed')
    assert terminal_worker('complete')
    assert terminal_worker('stopped')
    assert not terminal_worker('running')
    assert not terminal_worker('unknown')


def test_journal_code_is_after_pci_address_and_repeated_messages_are_one_incident():
    from scripts.analyze_gpu_journal import analyze
    def event(message, second, boot='a'):
        return dict(MESSAGE=message, __REALTIME_TIMESTAMP=str(second*1000000), _BOOT_ID=boot)
    rows = [event('NVRM: loading driver old', 1), event('NVRM: GPU at PCI:0000:01:00: GPU-aaaa-bbbb', 2),
            event('NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.', 10),
            event('NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.', 11),
            event('NVRM: Xid (PCI:0000:01:00): 154, Node Reboot Required', 12),
            event('NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.', 90, 'b')]
    result = analyze(rows, 'GPU-aaaa-bbbb')
    assert result['xid_counts'] == {'79': 3, '154': 1}
    assert len(result['xid79_clusters']) == 2
    assert result['xid79_clusters'][0]['xid79_messages'] == 2
    assert result['xid79_clusters'][0]['same_physical_gpu_uuid_confirmed'] is True
    assert result['xid79_clusters'][1]['same_physical_gpu_uuid_confirmed'] is None
    assert result['retained_boots'] == 2

    # Same boot and another PCI device must not establish the failing card's identity.
    other = [event('NVRM: GPU at PCI:0000:02:00: GPU-aaaa-bbbb', 1),
             event('NVRM: GPU at PCI:0000:01:00: GPU-cccc-dddd', 2),
             event('NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.', 3)]
    assert analyze(other, 'GPU-aaaa-bbbb')['events'][0]['same_physical_gpu_uuid_confirmed'] is False
