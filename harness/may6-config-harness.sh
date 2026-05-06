#!/usr/bin/env bash
# Run all 4 phases on a launched-and-ready container.
# Requires server already serving on PORT=11435 with model qwen3.6-27b
set -euo pipefail
CONFIG_LABEL="${1:?need config label}"
KV_BUDGET="${2:?need KV budget}"
OUT_DIR="${3:?need OUT_DIR}"

mkdir -p "$OUT_DIR"

PORT=11435
BENCH=/home/josh/qwen-vllm-test/llm-inference-bench/.venv/bin/python
SCRIPT=/home/josh/qwen-vllm-test/llm-inference-bench/llm_decode_bench.py
HARNESS=/home/josh/qwen-vllm-test/bench/stress-harness/stress_harness.py
PROBLEMS_DIR=/home/josh/qwen-vllm-test/bench/stress-harness/problems

START=$(date +%s)
echo ""
echo "═══════════════════════════════════════════════"
echo "  CONFIG: $CONFIG_LABEL"
echo "  Started: $(date +%H:%M:%S)"
echo "  KV budget: $KV_BUDGET"
echo "  Output: $OUT_DIR"
echo "═══════════════════════════════════════════════"

# ---------- Phase 1: Functional gates ----------
echo ""
echo "[$(date +%H:%M:%S)] PHASE 1: Functional gates (Fibonacci 5x, tool, reasoning, multi-turn)"
GATE_LOG="$OUT_DIR/gates.log"
python3 - <<PY 2>&1 | tee "$GATE_LOG"
import requests, sys
URL = 'http://localhost:11435/v1/chat/completions'; H = {'Content-Type': 'application/json'}
def ask(msgs, max_tokens=4096, tools=None):
    p = {'model':'qwen3.6-27b','messages':msgs,'temperature':0.0,'max_tokens':max_tokens}
    if tools: p['tools']=tools; p['tool_choice']='auto'
    r = requests.post(URL,headers=H,json=p,timeout=300).json()
    return r['choices'][0]['message']

results = {}
# Gate 1: Fibonacci 5x
try:
    ok = 0
    for i in range(5):
        m=ask([{'role':'user','content':'Output the first 10 Fibonacci numbers as a comma-separated list (start: 1, 1).'}])
        c=(m.get('content') or '').strip()
        if '1, 1, 2, 3, 5, 8, 13, 21, 34, 55' in c: ok += 1
    results['fib_5x'] = (ok, 5)
    print(f"Gate 1 (Fibonacci 5x): {ok}/5 -> {'PASS' if ok==5 else 'FAIL'}")
except Exception as e:
    print(f"Gate 1: EXCEPTION {e}"); results['fib_5x'] = (0, 5)

# Gate 2: Tool
try:
    m=ask([{'role':'user','content':'What is the current weather in Tokyo? Use the tool.'}],
        tools=[{'type':'function','function':{'name':'get_weather','description':'Get weather','parameters':{'type':'object','properties':{'city':{'type':'string'}},'required':['city']}}}])
    tcs=m.get('tool_calls') or []
    g2 = any(tc['function']['name']=='get_weather' and 'tokyo' in tc['function']['arguments'].lower() for tc in tcs)
    results['tool_call'] = g2
    print(f"Gate 2 (Tool call): {'PASS' if g2 else 'FAIL'}")
except Exception as e:
    print(f"Gate 2: EXCEPTION {e}"); results['tool_call'] = False

# Gate 3: Reasoning
try:
    m=ask([{'role':'user','content':'What is 47 times 83? Show the result as a number only on the last line.'}], 8192)
    c=(m.get('content') or '').strip()
    g3 = '3901' in c
    results['reasoning_47x83'] = g3
    print(f"Gate 3 (47x83=3901): {'PASS' if g3 else 'FAIL'}")
except Exception as e:
    print(f"Gate 3: EXCEPTION {e}"); results['reasoning_47x83'] = False

# Gate 4: Multi-turn
try:
    msgs=[{'role':'user','content':'Imagine the temperature in Tokyo is 28C. Just acknowledge.'}]
    t1=ask(msgs,2048); t1c=(t1.get('content') or '').strip(); msgs.append({'role':'assistant','content':t1c})
    msgs.append({'role':'user','content':'Now imagine Berlin is at 18C. Just acknowledge.'})
    t2=ask(msgs,2048); t2c=(t2.get('content') or '').strip(); msgs.append({'role':'assistant','content':t2c})
    msgs.append({'role':'user','content':'Which of the two cities I mentioned is warmer? Answer in one short sentence.'})
    t3=ask(msgs,4096); t3c=(t3.get('content') or '').strip()
    g4 = 'tokyo' in t3c.lower() and 'warm' in t3c.lower()
    results['multi_turn'] = g4
    print(f"Gate 4 (multi-turn): {'PASS' if g4 else 'FAIL'} -- T3: {t3c[:80]!r}")
except Exception as e:
    print(f"Gate 4: EXCEPTION {e}"); results['multi_turn'] = False

import json
total = sum([results['fib_5x'][0]==5, results['tool_call'], results['reasoning_47x83'], results['multi_turn']])
print(f"\nGates: {total}/4")
with open('$OUT_DIR/gates.json','w') as f:
    json.dump({'results':results,'gates_passed':total,'gates_total':4}, f, indent=2)
PY

# ---------- Phase 2: Throughput matrix ----------
echo ""
echo "[$(date +%H:%M:%S)] PHASE 2: Throughput matrix (9 cells, N=2)"
mkdir -p "$OUT_DIR/throughput-matrix/runs"

declare -a CELLS=(
  "1 0" "1 32000" "1 131072"
  "2 0" "2 32000" "2 131072"
  "4 0" "4 32000" "4 131072"
)

for CELL in "${CELLS[@]}"; do
  CONC=$(echo $CELL | cut -d' ' -f1)
  CTX=$(echo $CELL | cut -d' ' -f2)
  LBL="c${CONC}_ctx${CTX}"
  for run in 1 2; do
    OUT_RUN="$OUT_DIR/throughput-matrix/runs/${LBL}_run${run}"
    mkdir -p "$OUT_RUN"
    if [ -f "$OUT_RUN/results.json" ]; then continue; fi
    if ! curl -s -m 3 http://localhost:$PORT/v1/models 2>/dev/null | grep -q 'Qwen3.6-27B'; then
      echo "  CONTAINER DOWN at $LBL run$run"; break 2
    fi
    "$BENCH" "$SCRIPT" \
      --host localhost --port $PORT --model Qwen3.6-27B \
      --concurrency $CONC --contexts $CTX \
      --duration 30 --decode-warmup-seconds 10 \
      --kv-budget $KV_BUDGET --skip-prefill \
      --display-mode plain --no-hw-monitor \
      --output "$OUT_RUN/results.json" > "$OUT_RUN/bench.log" 2>&1 || echo "    run$run FAILED"
    sleep 1
  done
  python3 - <<PY
import json, glob, statistics
runs=[]
for r in sorted(glob.glob('$OUT_DIR/throughput-matrix/runs/${LBL}_run*/results.json')):
    try:
        v=json.load(open(r))['results'][0]['aggregate_tps']
        if v>5: runs.append(v)
    except: pass
if runs:
    m=statistics.mean(runs); s=statistics.stdev(runs) if len(runs)>1 else 0
    print(f"  $LBL: {m:.1f}±{s:.2f}")
else: print(f"  $LBL: NO DATA")
PY
done

# ---------- Phase 3: HumanEval stress ----------
echo ""
echo "[$(date +%H:%M:%S)] PHASE 3: HumanEval stress (164 problems, c=8)"
python3 "$HARNESS" \
  --url http://localhost:11435/v1/chat/completions \
  --model qwen3.6-27b \
  --config-label "$CONFIG_LABEL" \
  --benchmark humaneval \
  --problems-file "$PROBLEMS_DIR/humaneval.jsonl" \
  --output "$OUT_DIR/humaneval.jsonl" \
  --concurrency 8 \
  --max-tokens 8192 \
  --request-timeout 600 2>&1 | tail -10

# ---------- Phase 4: MBPP stress ----------
echo ""
echo "[$(date +%H:%M:%S)] PHASE 4: MBPP stress (257 problems, c=8)"
python3 "$HARNESS" \
  --url http://localhost:11435/v1/chat/completions \
  --model qwen3.6-27b \
  --config-label "$CONFIG_LABEL" \
  --benchmark mbpp \
  --problems-file "$PROBLEMS_DIR/mbpp.jsonl" \
  --output "$OUT_DIR/mbpp.jsonl" \
  --concurrency 8 \
  --max-tokens 8192 \
  --request-timeout 600 2>&1 | tail -10

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "═══════════════════════════════════════════════"
echo "  CONFIG $CONFIG_LABEL DONE in ${ELAPSED}s ($((ELAPSED/60)) min)"
echo "═══════════════════════════════════════════════"
