# nsorangym_gym

Gymnasium wrapper for ns-O-RAN style control (file + semaphore) against ns-3.

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Baseline (mmWave/LTE Traffic Steering / Energy Saving) |
| `5gnr_gym` | **5G LENA (NR)** integration work with `NSORAN_5GLENA` |

## Modes (ns-3 LENA)

| Mode | Flags | Behavior |
|------|--------|----------|
| **Logging only** (default) | `useSemaphores=0`, empty `controlFileName` | Writes `uemeas`/`cellmeas` (and traces). No gym needed. |
| **Gym** | `useSemaphores=1` + `controlFileName=ts_actions_for_ns3.csv` | POSIX semaphores + control CSV; needs `nsorangym_gym` agent. |

Logging-only example:

```bash
cd ~/ns-3-dev
./ns3 run "lena-oran-test --simTime=2 --useSemaphores=0"
```

Gym mode hangs in `sem_wait` until the Python env posts the control semaphore.
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
