#!/bin/zsh
# V9 Jarvis v2 pipeline: data -> 600-step annealing -> evals -> bundle export.
# Run from the repo root:  nohup scripts/run_jarvis_v2.sh > logs/jarvis_v2.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=/Volumes/Lexar/Projects/llm-eval/results
step() { echo "[$(date +%H:%M:%S)] $*"; }

step "1/6 data"
$PY scripts/build_tools_data.py --out data/sft/tools_v2/tools.jsonl --count 6000
$PY scripts/reasoning_gen.py --out data/sft/reasoning_v1/reasoning.jsonl --count 3000
$PY scripts/build_jarvis_v9.py --tokenizer data/v8/tokenizer.json \
  --output-dir data/sft/packed_v9_jarvis2 --tools data/sft/tools_v2/tools.jsonl \
  --tools-count 6000 --extra data/sft/reasoning_v1/reasoning.jsonl

step "2/6 training (600 steps from anneal_a)"
mkdir -p runs/sft_v9_jarvis2
$PY -m tinyllm.sft_train \
  --base-config runs/pretrain_v9/config.json \
  --base-checkpoint runs/sft_v9_anneal_a/checkpoints/best \
  --train-tokens data/sft/packed_v9_jarvis2/train.tokens.bin --train-mask data/sft/packed_v9_jarvis2/train.mask.bin \
  --val-tokens data/sft/packed_v9_jarvis2/val.tokens.bin --val-mask data/sft/packed_v9_jarvis2/val.mask.bin \
  --replay-bin data/v9/pretrain/tokenized/train.bin --replay-ratio 0.05 \
  --output-dir runs/sft_v9_jarvis2 --steps 600 --micro-batch 4 --grad-accum 8 --lr 1e-5 \
  --weight-decay 0.01 --grad-clip 1.0 --eval-interval 100 --eval-batches 20 --save-interval 100 \
  --log-interval 25 --seed 42 --dtype bfloat16

step "3/6 reasoning bench (held-out, reformulated)"
$PY scripts/bench_reasoning.py --backend tinyllm_run --model runs/sft_v9_anneal_a:best --json-out $R/reasoning_anneal_a.json > $R/reasoning_anneal_a.txt 2>&1
$PY scripts/bench_reasoning.py --backend tinyllm_run --model runs/sft_v9_jarvis:best --tools --json-out $R/reasoning_jarvis_v1.json > $R/reasoning_jarvis_v1.txt 2>&1
$PY scripts/bench_reasoning.py --backend tinyllm_run --model runs/sft_v9_jarvis2:best --tools --json-out $R/reasoning_jarvis_v2.json > $R/reasoning_jarvis_v2.txt 2>&1

step "4/6 neutral bench + regression"
$PY scripts/bench_neutral.py --backend tinyllm_run --model runs/sft_v9_jarvis2:best --tools --max-tokens 160 --json-out $R/neutral_jarvis_v2_tools.json > $R/neutral_jarvis_v2_tools.txt 2>&1
$PY scripts/bench_neutral.py --backend tinyllm_run --model runs/sft_v9_jarvis:best --tools --max-tokens 160 --json-out $R/neutral_jarvis_v1_tools_160.json > $R/neutral_jarvis_v1_tools_160.txt 2>&1
$PY scripts/chat_regression.py --run-dir runs/sft_v9_jarvis2 --seed 42 > $R/regression_jarvis_v2.txt 2>&1 || true

step "5/6 export bundle"
S=$(mktemp -d)
$PY -m tinyllm.quantize --run-dir runs/sft_v9_jarvis2 --checkpoint best --bits 8 --out $S/q8 > /dev/null
rm -rf exports/tinyllm-v9-jarvis2
mkdir -p exports/tinyllm-v9-jarvis2/tinyllm-v9
cp exports/tinyllm-v9-jarvis/{run.sh,server.py,requirements.txt,README.md} exports/tinyllm-v9-jarvis2/
cp -R exports/tinyllm-v9-jarvis/tinyllm exports/tinyllm-v9-jarvis2/
cp src/tinyllm/tools.py exports/tinyllm-v9-jarvis2/tinyllm/tools.py
$PY scripts/fp16_embeddings.py $S/q8/model.safetensors exports/tinyllm-v9-jarvis2/tinyllm-v9/model.safetensors
cp $S/q8/config.json $S/q8/tokenizer.json exports/tinyllm-v9-jarvis2/tinyllm-v9/
rm -rf $S

step "6/6 summary"
{
  for f in reasoning_anneal_a reasoning_jarvis_v1 reasoning_jarvis_v2 neutral_jarvis_v1_tools_160 neutral_jarvis_v2_tools; do
    echo "== $f"; grep -E "^  [a-zé_]+ +[0-9]+/[0-9]+|Score" $R/$f.txt
  done
  echo "== regression v2"; grep -E "Score" $R/regression_jarvis_v2.txt
  echo "== val_loss v2"; grep '"val_loss"' runs/sft_v9_jarvis2/log.jsonl | cut -c1-70
} > $R/jarvis2_summary.txt
echo DONE >> $R/jarvis2_summary.txt
step "done"
