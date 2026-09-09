# nsorangym_gym

Gymnasium wrapper for ns-O-RAN style control (file + semaphore) against ns-3.

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Baseline (mmWave/LTE Traffic Steering / Energy Saving) |
| `5gnr_gym` | **5G LENA (NR)** integration work with `NSORAN_5GLENA` |

## LENA / 5G NR notes (`5gnr_gym`)

Target ns-3 stack:

- ns-3: https://github.com/jaewook2/NSORAN_NS3 (`ns-3.46`)
- NR module: https://github.com/jaewook2/NSORAN_5GLENA (`5g-lena-v4.1.1-gym`)
- ORAN interface: https://github.com/jaewook2/nsoran_update

Control path (same as wineslab):

1. Gym writes actions to `ts_actions_for_ns3.csv` (`timestamp,ueId,nrCellId`)
2. NR `NrGnbNetDevice::ReadControlFile` parses the file
3. Applying HO inside ns-3 is **stubbed / commented** for now (parse + log only)
4. KPM / semaphore sync with LENA file names is WIP

Stub environment: `LenaTrafficSteeringEnv` (see `src/environments/lena_ts_env.py`).

## Install

```bash
hatch build
pip3 install dist/*.tar.gz
# or editable
pip3 install -e .
```

## Examples

```bash
# existing mmWave/LTE TS
python3 examples/traffic_steering.py --ns3_path /path/to/ns-o-ran-ns3-mmwave

# LENA stub (needs matching scratch scenario name later)
python3 examples/lena_traffic_steering.py --ns3_path /path/to/ns-3-dev
```
