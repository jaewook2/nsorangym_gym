# Toy example

설치와 파라미터는 [README.md](README.md)에 있습니다. 이 문서는 학습 없이 예제를 한 번 실행하고, 화면에 찍힌 줄과 결과 파일을 읽는 방법입니다.

두 예제는 다릅니다.

| 스크립트 | 제어 | 행동이 의미하는 것 |
| --- | --- | --- |
| `examples/duty.py` | 듀티 사이클 | 데이터 심볼 비율. 바로 적용됨 |
| `examples/asm.py` | ASM 상태기계 | 목표 상태. ns-3가 부품을 순서대로 바꿈 |

공통 준비입니다. `NS3_PATH`와 가상환경은 README의 설치를 끝낸 뒤입니다.

```bash
source ~/nsorangym/nsorangym_gym/venv/bin/activate
export NS3_PATH=~/nsorangym/nsorangym_mmwave
export NS3_SKIP_BUILD=1
cd ~/nsorangym/nsorangym_gym
cp src/environments/scenario_configurations/asm_use_case.json /tmp/toy_lab.json
```

`/tmp/toy_lab.json`에서 수업용으로 다음만 바꿉니다. 값은 한 칸짜리 리스트입니다.

```json
"nMmWaveEnb": [2],
"nUe": [2],
"ues": [2],
"simTime": [0.4],
"indicationPeriodicity": [0.1]
```

셀 2가 왼쪽 mmWave, 셀 3이 오른쪽 mmWave입니다. 시작 후 0.5초는 두 셀을 켠 채로 두는 warmup이라, 첫 관측 시각은 약 600 ms입니다.

## 1. 듀티 사이클

```bash
python examples/duty.py \
  --config /tmp/toy_lab.json \
  --ns3_path "$NS3_PATH" \
  --kind duty_mode \
  --num_steps 2 \
  --output_folder /tmp/duty_toy
```

`--kind on_off`는 100%와 TurnOff만 오갑니다. 기본값 `duty_mode`는 100, 75, 50, 25, 0%, TurnOff를 한 스텝에 한 단계만 바꿉니다.

정상 출력은 이런 형태입니다.

```text
Launch reset done
kind=duty_mode sequential=True cells [2, 3] ues=2
Step 1 cell 2 duty 1 (100% mmWave data), cell 3 duty 1 (100% mmWave data) done
  commanded={2: 1, 3: 1} applied={2: 1, 3: 1} reward=0.347 match=True
Mode match 2/2
```

| 필드 | 의미 |
| --- | --- |
| `kind=duty_mode` | 심볼 듀티 제어. ASM 상태기계가 아님 |
| `commanded` | Gym이 지시한 듀티 레벨 |
| `applied` | ns-3가 그 스텝에 실제로 둔 레벨. 듀티는 기다리지 않고 바로 바뀜 |
| `match=True` | 지시와 적용이 같음 |
| `appKbps` | 사용자 트래픽 처리량 |
| `pMmw`, `pLte` | mmWave와 LTE 전력 모델 |
| `reward` | 에너지 효율 보상. `--omega`를 주지 않으면 만족도는 보상 밖 |
| `Mode match 2/2` | 두 스텝 모두 지시가 바로 적용됨. 하나라도 다르면 프로세스가 실패로 끝남 |

레벨 6이면 그 셀의 사용자는 LTE로 넘어가고 `nUeLte`와 LTE 처리량이 늘어납니다.

## 2. ASM 상태기계

```bash
python examples/asm.py \
  --config /tmp/toy_lab.json \
  --ns3_path "$NS3_PATH" \
  --num_steps 2 \
  --output_folder /tmp/asm_toy
```

이 스크립트에는 `--kind`가 없습니다. 행동은 목표가 됩니다. 1은 Active, 2는 ASM1, 3은 ASM2, 4는 ASM3, 5는 ASM4, 6은 OFF입니다.

짧은 실행에서 나온 실제 두 스텝입니다.

```text
kind=asm cells [2, 3] ues=2
Step 1 cell 2 target 3 (ASM2), cell 3 target 4 (ASM3) done
  target={2: 3, 3: 4} state={2: (0, 3), 3: (0, 4)} applied={2: 3, 3: 4}
  asmPowerW=0.853 reward=0.165
Step 2 cell 2 target 5 (ASM4), cell 3 target 6 (OFF) done
  target={2: 5, 3: 6} state={2: (1, 5), 3: (1, 6)} applied={2: 3, 3: 4}
  asmPowerW=0.043 reward=0.000
ASM steps 2 target_observed=True
```

| 필드 | 의미 |
| --- | --- |
| `target=` | 이번 스텝에 Gym이 보낸 목표 |
| `state={셀: (위상, 목표)}` | 위상 0은 안정, 1은 내리는 중, 2는 켜는 중 |
| `applied` | 지금 도달해 있는 상태. 전이가 끝나면 목표와 같아짐 |
| `asmPowerW` | 두 셀의 ASM 부품 전력 합 |
| `target_observed=True` | 관측에 목표 one-hot이 있음. 없으면 상태기계가 켜지지 않은 것 |

첫 스텝은 PA와 RF, 베이스밴드를 끄는 짧은 전이라 100 ms 안에 ASM2, ASM3에 도착했습니다. `state`의 위상이 0이고 `applied`가 목표와 같습니다.

둘째 스텝의 ASM4와 OFF는 제어 블록과 전원까지 내려가므로 0.5초 단위입니다. 100 ms 시점에는 위상 1, 즉 아직 내리는 중이고 `applied`는 이전 상태 3과 4에 남아 있습니다. 이건 실패가 아닙니다. 전이가 끝나기 전에 다른 목표를 주면 ns-3는 그 목표를 무시합니다.

데이터는 안정 상태의 Active에서만 나갑니다. 둘째 스텝의 `appKbps=0`은 두 셀 모두 Active가 아니기 때문입니다.

## 3. 결과 파일

각 실행의 `sim_path` 디렉터리를 봅니다.

| 파일 | 듀티 | ASM |
| --- | --- | --- |
| `AsmActions.txt` | 지시한 듀티 레벨 | 지시한 목표 번호 |
| `asm-path-tput.csv` | LTE와 mmWave 처리량 | 처리량과 `asmState`, `asmPowerW` |
| `asm-mode-changes.txt` | 레벨이 바뀐 시각 | 전이가 끝난 시각 |
| `stderr` | 적용된 레벨 | `ASM target 1 -> 3`, `phase`, `delayMs` |

`stderr`에서 `ASM target`이 보이면 상태기계입니다. 듀티 실행에는 이 줄이 없고, 레벨이 바로 바뀝니다.

## 4. 학습으로 넘어갈 때

예제가 `ASM steps` 또는 `Mode match`까지 끝나면 같은 JSON으로 PPO를 돌립니다.

```bash
python examples/asm_ppo.py --config /tmp/toy_lab.json --ns3_path "$NS3_PATH" --n_envs 1 --n_steps 8 --batch_size 8 --no_until_converge --total_timesteps 32 --n_eval_episodes 1
python examples/duty_ppo.py --config /tmp/toy_lab.json --ns3_path "$NS3_PATH" --kind duty_mode --n_envs 1 --n_steps 8 --batch_size 8 --no_until_converge --total_timesteps 32 --n_eval_episodes 1
```

학습 예산, 보상 `--omega`, 시나리오 키 설명은 README에 있습니다.
