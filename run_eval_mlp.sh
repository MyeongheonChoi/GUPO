#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="outputs"

# 사용할 GPU 리스트 (예: 0번, 1번)
GPUS=(0 1)

# GPU별로 현재 돌고 있는 PID를 저장할 associative array
declare -A GPU_PIDS

# ckpt 디렉토리들 순회 (예: outputs/*/epoch-1, epoch-2 ...)
for ckpt_dir in "$BASE_DIR"/gupo_*/epoch-*; do
  [ -d "$ckpt_dir" ] || continue

  # 이 ckpt_dir를 어느 GPU에 올릴지 찾는 루프
  while true; do
    for gpu in "${GPUS[@]}"; do
      pid="${GPU_PIDS[$gpu]-}"

      # pid 가 비어있지 않고, 아직 프로세스가 살아있으면 이 GPU는 바쁨
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        continue
      fi

      # 여기까지 왔으면 이 GPU는 놀고 있음 → 이 GPU에 작업 올림
      echo "========== GPU ${gpu} 에서 실행: ${ckpt_dir} =========="
      CUDA_VISIBLE_DEVICES="${gpu}" uv run evaluate.py --checkpoint_dir "${ckpt_dir}" &
      GPU_PIDS[$gpu]=$!   # 방금 실행한 백그라운드 프로세스 PID 저장
      # 이 ckpt_dir는 할당했으니 다음 ckpt_dir로
      break 2
    done

    # 모든 GPU가 바쁘면 1초 쉬고 다시 확인
    sleep 1
  done
done

echo "모든 job 제출 완료. 남은 job들이 끝나길 기다리는 중..."
wait
echo "끝!"
