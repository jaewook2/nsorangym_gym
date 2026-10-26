# 듀티 사이클 제어 실습과 AI 학습

이 문서는 `nsorangym_gym`의 `asm_duty` 브랜치용입니다. Gym이 100 ms마다 mmWave 셀의 데이터 심볼 비율을 바꾸거나 셀을 끄고, ns-3가 처리량과 에너지 지표를 돌려줍니다.

여기서 숫자 1–6은 듀티 레벨입니다. PA, RF, 베이스밴드를 순서대로 끄는 ASM 상태기계(Active, ASM1–ASM4, OFF)는 이 브랜치에 없습니다. 그 제어는 `asm_state` 브랜치입니다.

코드와 설정 파일은 예전 이름인 `asmMode`, `--kind asm`을 그대로 씁니다. `asm_duty`에서 그 값은 듀티 레벨입니다.

구성은 다음 세 저장소입니다. mmWave의 `contrib`는 비어 있으므로 nsoran을 그 안에 받습니다.

| 역할 | 저장소 | 브랜치 |
| --- | --- | --- |
| E2SIM | https://github.com/jaewook2/nsorangym_e2sim | `main` |
| mmWave | https://github.com/jaewook2/nsorangym_mmwave | `asm_duty` |
| nsoran | https://github.com/jaewook2/nsorangym_nsoran | `main` |
| gym | https://github.com/jaewook2/nsorangym_gym | `asm_duty` |

아래 경로에서 `~/nsorangym`은 실습 폴더입니다. 다른 위치를 써도 됩니다.

## 1. 설치

### 시스템 패키지

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential git cmake libsctp-dev autoconf automake libtool bison flex \
  libboost-all-dev g++ python3 python3-pip python3-venv python3-dev \
  pkg-config libeigen3-dev sqlite3
```

### E2SIM

```bash
mkdir -p ~/nsorangym && cd ~/nsorangym
git clone -b main https://github.com/jaewook2/nsorangym_e2sim.git
cd nsorangym_e2sim/e2sim
mkdir -p build
./build_e2sim.sh 2
```

이 스크립트는 `e2sim` 패키지를 시스템에 설치합니다. nsoran은 헤더를 `/usr/local/include/e2sim`에서 찾습니다. 설치 위치가 다르면 `contrib/oran-interface/CMakeLists.txt`의 `SEARCH_PATHS`를 그 위치로 바꿉니다.

### ns-3와 nsoran

```bash
cd ~/nsorangym
git clone -b asm_duty https://github.com/jaewook2/nsorangym_mmwave.git
git clone -b main https://github.com/jaewook2/nsorangym_nsoran.git \
  nsorangym_mmwave/contrib/oran-interface
cd nsorangym_mmwave
./ns3 configure --enable-examples --disable-gtk --disable-werror
./ns3 build scenario-asm-verify
```

실습 시나리오 이름은 `scenario-asm-verify`입니다. 기본 배치는 가운데 LTE eNB, 왼쪽과 오른쪽 mmWave gNB입니다.

### Gym 파이썬 환경

```bash
cd ~/nsorangym
git clone -b asm_duty https://github.com/jaewook2/nsorangym_gym.git
cd nsorangym_gym
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -e .
pip install "stable-baselines3>=2.3" torch
```

`pip install -e .`가 Gymnasium, NumPy, pandas, posix-ipc, sem을 설치합니다. PPO와 DQN은 stable-baselines3를 사용합니다.

매번 새 터미널에서는 다음만 다시 하면 됩니다.

```bash
source ~/nsorangym/nsorangym_gym/venv/bin/activate
export NS3_PATH=~/nsorangym/nsorangym_mmwave
export NS3_SKIP_BUILD=1
cd ~/nsorangym/nsorangym_gym
```

`NS3_SKIP_BUILD=1`은 이미 빌드된 ns-3를 에피소드마다 다시 빌드하지 않게 합니다.

## 2. 듀티 레벨

셀마다 레벨은 1부터 6입니다. 바로 적용되며, 부품을 하나씩 끄는 전이 시간은 없습니다.

| 레벨 | 동작 |
| --- | --- |
| 1 | mmWave 데이터 심볼 100% |
| 2 | 75% |
| 3 | 50% |
| 4 | 25% |
| 5 | 0%. 셀은 켜져 있음 |
| 6 | TurnOff. 사용자는 LTE로 넘어감 |

제어 종류는 두 가지입니다.

| `--kind` | 행동 | 동작 |
| --- | --- | --- |
| `on_off` | 셀마다 0 또는 1 | 0은 레벨 1, 1은 레벨 6. 모든 셀을 동시에 끄지 않음 |
| `asm` 또는 `duty` | 셀마다 0–5 | 레벨 1–6. 한 스텝에 ±1만 변경. 모든 셀을 동시에 끄지 않음 |

`examples/asm.py`의 기본값은 `--kind asm`입니다. 이 브랜치에서 그 값은 위 표의 순차 듀티입니다.

시작 후 0.5초는 모든 mmWave 셀을 레벨 1로 두는 warmup입니다. JSON의 `simTime`에 이 시간과 한 주기가 더해져 실제 시뮬레이션이 조금 더 길어집니다.

## 3. 실습

먼저 짧은 무작위 제어로 루프가 도는지 확인합니다. 수업용으로는 UE 수와 시간을 줄입니다.

```bash
cd ~/nsorangym/nsorangym_gym
cp src/environments/scenario_configurations/asm_use_case.json /tmp/duty_lab.json
```

`/tmp/duty_lab.json`에서 다음만 바꿉니다. 값은 반드시 한 칸짜리 리스트로 둡니다.

```json
"nMmWaveEnb": [2],
"nUe": [4],
"ues": [4],
"simTime": [1.0],
"indicationPeriodicity": [0.1]
```

켜기/끄기만 보려면 `on_off`를 씁니다.

```bash
python examples/asm.py \
  --config /tmp/duty_lab.json \
  --ns3_path "$NS3_PATH" \
  --kind on_off \
  --num_steps 4 \
  --output_folder examples/output_duty_lab
```

75%, 50%처럼 중간 듀티를 보려면 `--kind duty`로 바꿉니다.

정상이면 `Launch reset done` 다음에 스텝마다 `reward=`가 출력되고, 마지막에 `Mode match 4/4`가 나옵니다. 결과 폴더 안의 시뮬레이션 디렉터리에서 다음 파일을 봅니다.

| 파일 | 내용 |
| --- | --- |
| `AsmActions.txt` | Gym이 내린 `timestamp,cellId,asmMode` |
| `asm-path-tput.csv` | LTE와 mmWave PDCP 처리량 |
| `cu-up-cell-*.txt` | 셀별 PDCP 지표 |
| `stderr` | ns-3 로그. `2:6`처럼 적용된 레벨이 찍힘 |

레벨 6이 적용되면 해당 mmWave 셀의 트래픽이 LTE 처리량으로 이동합니다.

## 4. AI 학습

같은 JSON으로 PPO를 돌립니다. 처음에는 환경 1개와 짧은 예산으로 확인합니다.

```bash
python examples/asm_ppo.py \
  --config /tmp/duty_lab.json \
  --ns3_path "$NS3_PATH" \
  --kind on_off \
  --n_envs 1 \
  --n_steps 8 \
  --batch_size 8 \
  --no_until_converge \
  --total_timesteps 32 \
  --n_eval_episodes 1 \
  --output_folder examples/output_duty_ppo_lab
```

순차 듀티를 학습하려면 `--kind duty`를 씁니다. `--kind both`는 `on_off`와 순차 듀티를 이어서 학습합니다.

`n_steps * n_envs`는 `batch_size`로 나누어 떨어져야 합니다. 위 예는 8 × 1 = 8입니다.

학습이 끝나면 `examples/output_duty_ppo_lab` 아래에 다음이 있습니다.

| 경로 | 내용 |
| --- | --- |
| `on_off/ppo_on_off.zip` | 학습된 정책 |
| `on_off/vecnormalize.pkl` | 관측 정규화 |
| `on_off/checkpoints/` | 중간 체크포인트 |
| `tb/on_off/` | TensorBoard 로그 |

비교만 다시 할 때는 `--eval_only`를 붙입니다. 기본 `--n_envs 8`과 `--until_converge`는 ns-3를 여러 개 오래 실행하므로 수업 확인용으로는 쓰지 않습니다.

보상은 `--omega`로 섞습니다. 0이면 에너지 효율만, 1이면 사용자 만족도 하위 5%(`satP5`)만 사용합니다. 기본값은 0입니다. 에너지 모델은 듀티 비율에 따라 송신 전력을 줄입니다.

ON/OFF를 DQN으로 학습하려면 `examples/asm_dqn.py`를 같은 인자로 실행합니다. DQN은 셀별 행동을 하나의 정수로 합칩니다.

## 5. 파라미터

시나리오 값은 `src/environments/scenario_configurations/`의 JSON입니다. 기본 파일은 `asm_use_case.json`입니다. ns-3는 각 키의 첫 번째 원소만 읽습니다.

| 키 | 기본값 | 의미 |
| --- | --- | --- |
| `nMmWaveEnb` | 2 | mmWave gNB 수. 셀 ID는 2부터 시작 |
| `ues`, `nUe` | 30 | UE 수. 두 키를 같은 값으로 둠 |
| `gnbDistance` | 250 | LTE에서 mmWave gNB까지 거리 (m) |
| `uePlacement` | `cluster` | `cluster`는 gNB 주변 배치, `random`은 사각 영역 |
| `ueClusterRadius` | 25 | 클러스터 반지름 (m) |
| `rateLeft`, `rateRight` | `2Mbps`, `600kbps` | 클러스터 배치의 왼쪽, 오른쪽 UE 전송률 |
| `rateHigh`, `rateMid`, `rateLow` | `10Mbps`, `2Mbps`, `500kbps` | `random` 배치의 3단계 전송률 |
| `simTime` | 3 | 트래픽 시간 (초). warmup 0.5초가 추가됨 |
| `indicationPeriodicity` | 0.1 | 관측과 제어 주기 (초) |
| `configuration` | 2 | 0=850 MHz, 1=3.5 GHz, 2=28 GHz |
| `lteBandwidthRbs` | 6 | LTE RB 수. 6이면 1.4 MHz |
| `bufferSize` | 10 | RLC 버퍼 (MB) |
| `rlcAmEnabled` | 1 | 1이면 RLC AM |
| `asmMode` | 1 | 시작 듀티 레벨. 상태기계 모드가 아님 |
| `useSemaphores` | 1 | Gym 동기화. 실습에서는 1로 유지 |
| `controlFileName` | `asm_actions_for_ns3.csv` | Gym이 ns-3에 넘기는 제어 파일 |
| `scheduleAsmControl` | 0 | 0이면 파일에 미리 적어 둔 스케줄을 쓰지 않음 |
| `flushAsmControlFile` | 0 | 여러 gNB가 CSV 하나를 공유할 때 0 |

관측은 0에서 1 사이입니다. mmWave 셀마다 듀티 레벨 one-hot, UE 비율, 제공 트래픽 비율, 평균 만족도, RLC 점유율이 있고, 마지막에 LTE UE 비율, LTE 처리량 비율, LTE 듀티, 전체 트래픽, 평균 만족도, `satP5`, LTE RLC 점유율이 붙습니다.

## 6. 확인할 문제

- `Cannot find scenario-asm-verify`: `NS3_PATH`가 mmWave 저장소 루트가 아니거나, 그 안에서 `./ns3 build scenario-asm-verify`가 끝나지 않았습니다.
- `n_steps*n_envs must be divisible by batch_size`: 세 숫자를 다시 맞춥니다.
- 세마포어 대기 후 바로 종료: `useSemaphores`가 1인지, 이전 실험 프로세스가 남아 있지 않은지 확인합니다.
- ns-3 로그에 `Connecting to server at 10.244.0.240:36422`가 보일 수 있습니다. 실습 제어는 CSV와 세마포어로 이루어집니다. 긴 실행 중 이 접속 실패로 프로세스가 죽으면 강사에게 알립니다.
