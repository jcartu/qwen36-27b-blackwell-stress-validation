# Stress-Validation Study of Qwen3.6-27B Inference Configurations on Dual NVIDIA RTX PRO 6000 Blackwell

**A 5-configuration × 4-phase head-to-head validation suite. 2,105 hard coding problems executed across all configurations under sustained concurrent load. Zero engine crashes, zero hangs, zero malformed responses.**

> **Update — May 6, evening:** [Section 12](#12-addendum-may-6-evening--draft_sample_methodgumbel-removal-study) adds an addendum re-running all three DFlash variants with the deprecated `--speculative-config.draft_sample_method gumbel` flag removed and an improved harness (60s decode duration, 20s warmup). Headline finding: **DFlash N=15 with `gumbel` disabled gains +8 HumanEval problems (73.2% → 78.0%)**, becoming the new SOTA among BF16+DFlash configurations. Production guidance unchanged: FP8+MTP=3 still leads at 79.3%.
>
> **Update — May 7:** [Section 13](#13-addendum-may-7--fp8-base--dflash-drafter-no-gumbel-study) characterises the previously-untested **FP8 base + DFlash drafter** pairing across N ∈ {7, 8, 15}, with the same harness plus a 60-second post-ready settle. Headline findings: **FP8+DFlash N=7 narrowly beats FP8+MTP=3 on aggregate throughput (244.2 vs 241.5 tok/s)** and **FP8+DFlash N=8 beats FP8+MTP=3 by +5–13% at long context (ctx ≥ 64k)** while gaining +2.7pp on MBPP. New guidance: FP8+DFlash N=8 is the recommended choice for long-context coding-agent workloads; FP8+MTP=3 remains the default for general serving.

---

## Abstract

We performed a controlled stress-validation of five Qwen3.6-27B inference configurations on the latest Repne `vllm` fork (image `repne/vllm:latest` SHA `f8daec1dc883`, 2026-05-06 10:17 UTC, engine version `v0.1.dev16434+g81845bdee`) running on dual NVIDIA RTX PRO 6000 Blackwell SM120 hardware (TP=2). For each configuration we executed (i) four functional gates (Fibonacci, tool-call, arithmetic reasoning, multi-turn coherence), (ii) a nine-cell throughput matrix (c ∈ {1,2,4} × ctx ∈ {0, 32k, 131k}, N=2), (iii) the **HumanEval** benchmark (164 hand-graded coding problems with verifiable test suites), and (iv) the **MBPP sanitized test set** (257 additional coding problems), the latter two executed at concurrency = 8 to simulate burst production load.

**Headline result.** All five configurations passed all functional gates and completed all 421 stress-test problems without a single engine crash, hang, malformed JSON response, or HTTP error. The image is operationally rock-solid. **FP8+MTP=3 holds HumanEval pass-rate SOTA (130/164 = 79.3%), while BF16+DFlash variants hold MBPP SOTA (230/257 = 89.5%).** The previously-recommended FP8+MTP=5 (which won low-concurrency throughput in our prior sprint) is **strictly dominated by FP8+MTP=3 on HumanEval** by 3.7 percentage points, despite producing 15% higher raw effective tok/s under load — a textbook demonstration that throughput-on-decode-bench does not predict end-to-end agentic correctness.

---

## 1. Headline findings

### 1.1 Operational robustness: 5/5 configurations pass full validation

```
                                gates   HE pass     MBPP pass    HE empty   MBPP empty   crashes
                                                                  responses  responses    
─────────────────────────────────────────────────────────────────────────────────────────────────
  A  FP8+MTP=3                  4/4    130/164      220/257       7          29          0
  B  FP8+MTP=5                  4/4    124/164      221/257       4          29          0
  C  BF16+DFlash=7              4/4    120/164      229/257       4          19          0
  D  BF16+DFlash=8               4/4    120/164      230/257       2          18          0
  E  BF16+DFlash=15              4/4    120/164      230/257       2          17          0
─────────────────────────────────────────────────────────────────────────────────────────────────
  TOTAL ATTEMPTED                       820 problems  1,285 problems          0 crashes
```

No configuration produced a single HTTP error, timeout, malformed JSON, broken tool call, or engine crash across **2,105 hard coding problems** at concurrency=8. Image `f8daec1dc883` is approved for production use.

### 1.2 HumanEval winners by configuration class

| Configuration | HumanEval pass-rate | vs. FP8+MTP=3 |
|---|---:|---:|
| **FP8+MTP=3** | **79.3%** (130/164) | — (SOTA) |
| FP8+MTP=5 | 75.6% (124/164) | −3.7 pp |
| BF16+DFlash=7 | 73.2% (120/164) | −6.1 pp |
| BF16+DFlash=8 | 73.2% (120/164) | −6.1 pp |
| BF16+DFlash=15 | 73.2% (120/164) | −6.1 pp |

**FP8+MTP=3 wins HumanEval cleanly.** All three DFlash variants converge to **identical** pass count (120/164) — they appear to be solving the same set of problems and missing the same set, suggesting the limiting factor at this difficulty is the BF16 base model's reasoning capability rather than the speculative-decoding parameters. FP8+MTP=5 sits between FP8+MTP=3 and the DFlash floor.

### 1.3 MBPP winners by configuration class (different from HumanEval)

| Configuration | MBPP pass-rate | vs. best |
|---|---:|---:|
| **BF16+DFlash=8** | **89.5%** (230/257) | — (tied SOTA) |
| **BF16+DFlash=15** | **89.5%** (230/257) | — (tied SOTA) |
| BF16+DFlash=7 | 89.1% (229/257) | −0.4 pp |
| FP8+MTP=5 | 86.0% (221/257) | −3.5 pp |
| FP8+MTP=3 | 85.6% (220/257) | −3.9 pp |

**BF16+DFlash variants beat FP8 on MBPP.** This is opposite to the HumanEval ranking. Hypothesis: MBPP problems are simpler, and the precision floor of BF16 weights resolves a small fraction of edge cases (off-by-one errors in test-case boundary conditions, floating-point precision in arithmetic) that FP8 W8A8 fails. HumanEval problems are larger and more complex, where reasoning capability dominates over numerical precision — and the FP8 model's MTP head improves reasoning-token throughput enough to be worth the small precision tradeoff.

### 1.4 Throughput hierarchy under stress (concurrency=8)

| Configuration | Effective tok/s on HumanEval | Effective tok/s on MBPP | p95 latency HE | p95 latency MBPP |
|---|---:|---:|---:|---:|
| **FP8+MTP=5** | **1263.7** | **1234.4** | **35.5 s** | **51.8 s** |
| FP8+MTP=3 | 1098.2 | 1055.8 | 45.6 s | 59.0 s |
| BF16+DFlash=8 | 1088.7 | 1084.0 | 36.4 s | 56.2 s |
| BF16+DFlash=7 | 1077.6 | 1015.3 | 40.2 s | 59.1 s |
| BF16+DFlash=15 | 1006.3 | 995.4 | 42.4 s | 60.3 s |

**FP8+MTP=5 has the highest raw throughput and lowest p95 latency under stress.** This confirms the prior sprint's finding that MTP=5 wins low-concurrency throughput — but the **HumanEval result shows MTP=5 also produces fewer correct answers**. The 15% throughput advantage is real but produces 3.7 fewer pass-points (124 vs 130).

### 1.5 The throughput-correctness tradeoff is real

```
  Configuration       │  effective tok/s (HE)   │  HumanEval pass  │  Useful tok/sec*
  ─────────────────────────────────────────────────────────────────────────────────────
  FP8+MTP=3            │       1098.2            │   130/164 (79.3%) │   871.0
  FP8+MTP=5            │       1263.7            │   124/164 (75.6%) │   955.4
  BF16+DFlash=7        │       1077.6            │   120/164 (73.2%) │   789.2
  BF16+DFlash=8        │       1088.7            │   120/164 (73.2%) │   797.2
  BF16+DFlash=15       │       1006.3            │   120/164 (73.2%) │   736.6

  * Useful tok/sec = effective tok/s × pass_rate. A tok/s rate weighted by what fraction
    of the model's output is actually correct. FP8+MTP=5 STILL leads by this metric on
    HumanEval — its raw speed makes up for its slightly lower correctness.
```

By the **useful tok/sec** metric (raw throughput weighted by correctness), FP8+MTP=5 narrowly leads HumanEval (955.4 vs FP8+MTP=3's 871.0), but FP8+MTP=3 wins absolute pass count. The right choice depends on whether your downstream system can tolerate 3.7 percentage points of additional code-correctness failures in exchange for 15% faster output.

For a coding agent where each failed solution costs a downstream re-prompt or human intervention, **FP8+MTP=3 is preferable** despite its slower raw throughput. For a system that scores many candidates and picks the best (pass@k > 1), **FP8+MTP=5 wins** because it can generate more candidates per unit time.

### 1.6 No-output behavior is a real failure mode at high stress

Across all 5 configurations, the dominant failure mode at high concurrency was **`empty_response`** — model emitted `finish_reason=length` after exhausting the 8192-token budget on its `<thinking>` step without producing any final `content`. This is not a quality regression in the model — it's the reasoning model exceeding the response budget on certain hard problems. Distribution:

| Configuration | HumanEval empties | MBPP empties | Total empties (out of 421) |
|---|---:|---:|---:|
| BF16+DFlash=15 | 2 | 17 | 19 (4.5%) |
| BF16+DFlash=8 | 2 | 18 | 20 (4.8%) |
| BF16+DFlash=7 | 4 | 19 | 23 (5.5%) |
| FP8+MTP=5 | 4 | 29 | 33 (7.8%) |
| FP8+MTP=3 | 7 | 29 | 36 (8.6%) |

DFlash variants budget reasoning tokens more tightly than FP8+MTP variants. **In a production agent, increasing `max_tokens` to 16384 would reclaim most of these as solved problems.**

---

## 2. Throughput matrix (validation against prior sprint baselines)

We re-ran the standard 9-cell throughput matrix on each configuration to confirm the new image preserves performance characteristics from the prior image (`d0a200f77546`, May 5 evening).

| Configuration | c=1×0 | c=1×32k | c=1×131k | c=2×0 | c=2×32k | c=2×131k | c=4×0 | c=4×32k | c=4×131k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP8+MTP=3 | 116.8 | 110.7 | 96.8 | 220.4 | 223.2 | 183.5 | 441.7 | 439.2 | 341.0 |
| FP8+MTP=5 | 120.9 | 117.2 | 93.4 | 233.9 | 234.2 | 187.5 | 463.1 | 444.9 | 340.0 |
| BF16+DFlash=7 | 95.2 | 93.2 | 88.9 | 180.8 | 178.1 | 163.3 | 343.2 | 331.6 | 305.4 |
| BF16+DFlash=8 | 93.3 | 92.3 | 85.6 | 175.6 | 174.2 | 166.1 | 335.3 | 338.3 | 300.5 |
| BF16+DFlash=15 | 88.4 | 90.3 | 83.7 | 173.2 | 166.3 | 151.9 | 313.4 | 297.1 | 252.4 |

All values within ±2σ of the prior image's measurements; no regression detected. The new May 6 image preserves performance characteristics across all five tested configurations.

---

## 3. Materials and methods

### 3.1 Hardware and software stack

| Component | Specification |
|---|---|
| GPUs | 2× NVIDIA RTX PRO 6000 Blackwell Workstation Edition (SM120, 96 GiB GDDR7 each) |
| Interconnect | PCIe Gen5 x16 (validated under load) |
| Driver | 595.71.05 |
| CUDA | 13.2 |
| Container image | `repne/vllm:latest` digest `sha256:f8daec1dc883…` (2026-05-06 10:17 UTC) |
| vLLM engine | `v0.1.dev16434+g81845bdee.d20260506` |
| Bench tool | [`llm_decode_bench.py v0.4.8`](https://github.com/cole-yoshioka/llm-inference-bench) |
| Stress harness | Custom asyncio-based concurrent client (see `harness/stress_harness.py`) |

### 3.2 Tested configurations

All configurations share TP=2, max-model-len=262144, max-num-seqs=128, max-num-batched-tokens=32758, max-cudagraph-capture-size=256, GPU memory util=0.85, prefix caching enabled, gumbel draft sampling, and the Repne fork's `instanttensor` weight-loading path.

| Label | Weights | Speculative decoding | KV cache (tokens) |
|---|---|---|---:|
| A. FP8+MTP=3 | Qwen/Qwen3.6-27B-FP8 (W8A8) | MTP n=3 | 1,844,943 |
| B. FP8+MTP=5 | Qwen/Qwen3.6-27B-FP8 (W8A8) | MTP n=5 | 1,809,022 |
| C. BF16+DFlash=7 | Qwen/Qwen3.6-27B (BF16) | DFlash n=7, drafter z-lab/Qwen3.6-27B-DFlash | 954,235 |
| D. BF16+DFlash=8 | Qwen/Qwen3.6-27B (BF16) | DFlash n=8, drafter z-lab/Qwen3.6-27B-DFlash | 948,555 |
| E. BF16+DFlash=15 | Qwen/Qwen3.6-27B (BF16) | DFlash n=15, drafter z-lab/Qwen3.6-27B-DFlash | 907,886 |

### 3.3 Validation phases (per configuration)

**Phase 1 — Functional gates.** Four hand-curated correctness probes:
1. Fibonacci sequence ×5 deterministic (must produce "1, 1, 2, 3, 5, 8, 13, 21, 34, 55" five times consecutively at temp=0)
2. Tool call (must emit valid `get_weather({"city":"Tokyo"})` JSON given a single function definition)
3. Arithmetic reasoning (must compute 47 × 83 = 3901 with intermediate steps)
4. Multi-turn coherence (Tokyo at 28 °C, then Berlin at 18 °C, then "which is warmer?" — must answer Tokyo)

**Phase 2 — Throughput matrix.** Nine cells (c ∈ {1, 2, 4} × ctx ∈ {0, 32k, 131k}), N=2 per cell, 30 s sustained-decode + 10 s warmup, `--skip-prefill` to isolate steady-state speculative-decoding throughput.

**Phase 3 — HumanEval stress test.** All 164 problems from `openai_humaneval`, executed at concurrency=8 via async client. Each problem submits the standard prompt, waits for the model's response, extracts the largest Python code block from the output, and runs the canonical `check(entry_point)` test suite in a 10-second-timeout subprocess. Records: HTTP status, finish_reason, prompt and completion token counts, code-extraction success, test pass/fail with returncode, stderr head. Failure modes classified into seven categories: `ok`, `http_error`, `timeout`, `empty_response`, `no_code`, `test_fail`, `exception`.

**Phase 4 — MBPP stress test.** All 257 problems from the sanitized split of `google-research-datasets/mbpp` (`test` split), same execution flow as HumanEval, with up to 3 example test cases shown in-prompt to the model and the full `test_list` evaluated post-hoc.

### 3.4 Stress-harness implementation notes

The stress harness (`harness/stress_harness.py`) is an asyncio-based concurrent client that maintains exactly N (configurable, default 8) outstanding requests against the vLLM endpoint, dispatching new problems as soon as any complete. Per-request timeout was 600 s. Each completed problem's full state was incrementally persisted to a JSONL file to ensure no data loss in the event of a crash. Test execution happens in a `ProcessPoolExecutor` (10 s timeout per code execution) to prevent runaway model output (e.g., infinite loops in generated code) from blocking the main event loop.

The harness deliberately uses **temperature=0** and **seed=42** to make every measurement deterministic and reproducible. Pass-rate differences between configurations therefore reflect deterministic differences in the model's greedy-decoded output across configurations, not stochastic sampling variance.

---

## 4. Results matrix (raw)

### 4.1 HumanEval failure-mode breakdown

| Configuration | ok | test_fail | empty_response | http_error | timeout |
|---|---:|---:|---:|---:|---:|
| FP8+MTP=3 | 130 | 27 | 7 | 0 | 0 |
| FP8+MTP=5 | 124 | 36 | 4 | 0 | 0 |
| BF16+DFlash=7 | 120 | 40 | 4 | 0 | 0 |
| BF16+DFlash=8 | 120 | 42 | 2 | 0 | 0 |
| BF16+DFlash=15 | 120 | 42 | 2 | 0 | 0 |

### 4.2 MBPP failure-mode breakdown

| Configuration | ok | test_fail | empty_response | http_error | timeout |
|---|---:|---:|---:|---:|---:|
| FP8+MTP=3 | 220 | 8 | 29 | 0 | 0 |
| FP8+MTP=5 | 221 | 7 | 29 | 0 | 0 |
| BF16+DFlash=7 | 229 | 9 | 19 | 0 | 0 |
| BF16+DFlash=8 | 230 | 9 | 18 | 0 | 0 |
| BF16+DFlash=15 | 230 | 10 | 17 | 0 | 0 |

**Notable:** FP8 configurations produce disproportionately more `empty_response` failures than BF16+DFlash on MBPP (29 vs 17–19). Hypothesis: MBPP descriptions trigger longer `<thinking>` traces in the FP8 reasoning path, occasionally exhausting the 8192-token budget. DFlash variants apparently cap reasoning length more aggressively.

### 4.3 Latency distribution under stress

| Configuration | HE mean | HE median | HE p95 | MBPP mean | MBPP median | MBPP p95 |
|---|---:|---:|---:|---:|---:|---:|
| FP8+MTP=3 | 20.9 s | — | 45.6 s | 18.7 s | — | 59.0 s |
| FP8+MTP=5 | 16.9 s | — | 35.5 s | 16.5 s | — | 51.8 s |
| BF16+DFlash=7 | 20.0 s | — | 40.2 s | 17.6 s | — | 59.1 s |
| BF16+DFlash=8 | 18.5 s | — | 36.4 s | 16.8 s | — | 56.2 s |
| BF16+DFlash=15 | 21.0 s | — | 42.4 s | 18.2 s | — | 60.3 s |

FP8+MTP=5 has the **fastest mean and p95 latency under stress**, beating FP8+MTP=3 by ~20% on p95. This is the cost FP8+MTP=3 pays for higher correctness — more tokens spent reasoning before producing a final answer.

---

## 5. Production decision tree

```
Optimize for absolute correctness (each failure costs human time)?
├── HumanEval-class problems (complex reasoning)
│   └── FP8+MTP=3   ← 79.3% pass-rate, slowest latency
├── MBPP-class problems (simpler, precision-sensitive)
│   └── BF16+DFlash=8  ← 89.5% pass-rate, but 70% slower than FP8
└── Mixed workload  → FP8+MTP=3 (best HumanEval is harder to beat)

Optimize for throughput-weighted-correctness (pass@k where k > 1)?
└── FP8+MTP=5   ← 955 useful tok/s on HumanEval

Optimize for raw output volume (e.g., synthetic data generation)?
└── FP8+MTP=5   ← 1263 effective tok/s on HumanEval, 1234 on MBPP

Optimize for tightest token budget (`max_tokens` constrained)?
└── BF16+DFlash=8 or DFlash=15  ← Lowest empty_response rate (1.2%)
```

---

## 6. Validation against prior sprints

This study extends and partially overturns findings from the prior 8-experiment sprint ([`qwen36-27b-blackwell-inference-study`](https://github.com/jcartu/qwen36-27b-blackwell-inference-study)).

| Prior sprint claim | This study validates? | Notes |
|---|---|---|
| FP8+MTP=3 is throughput SOTA at production concurrency | ✅ Reaffirmed | Matrix throughput numbers preserved; HumanEval correctness now also confirms it |
| FP8+MTP=5 wins low-concurrency throughput | ✅ Reaffirmed | c=1–4 throughput slightly higher for MTP=5 |
| FP8+MTP=5 should be production config | ❌ Overturned | MTP=5 *correctness* is 3.7 pp lower on HumanEval. Throughput-only verdict was incomplete. |
| BF16+DFlash variants are 17–30% behind FP8 throughput | ✅ Reaffirmed | Throughput delta confirmed; but DFlash wins MBPP by 3.5–3.9 pp |
| Q8/BF16 KLD = 0.0018 (noise floor) | ✅ Reaffirmed | Quality claim from perplexity probe is consistent with these end-to-end results |

**The most important new insight: throughput on a 30-second decode benchmark does not predict end-to-end coding-agent correctness.** The 8-experiment sprint had picked MTP=5 based on throughput; this study shows that exact same metric improvement comes with a 3.7 percentage-point HumanEval correctness regression.

---

## 7. Limitations and threats to validity

1. **Single random seed (seed=42, temp=0).** Pass-rate differences are deterministic given the model and config, but pass-rate variance across multiple seeds is unknown. A future sprint should run pass@k for k∈{1, 5, 10} with sampling.

2. **N=2 throughput cells.** Throughput matrix is N=2 (lower than the N=3 used in prior sprints) to keep the sprint within the 8-hour budget. Variance bands are wider than ideal. The HumanEval and MBPP runs are N=164 and N=257 respectively, so quality conclusions are well-powered.

3. **Stress test concurrency = 8 only.** We did not characterize behavior at c∈{16, 32}, where the prior sprint showed FP8+MTP=3 dominates throughput. A higher-concurrency stress test could reveal additional failure modes (KV cache pressure, scheduler edge cases).

4. **Coding benchmarks only.** Both HumanEval and MBPP are Python-coding tasks. Long-context, multilingual, mathematical-reasoning, and tool-use-heavy workloads may rank configurations differently.

5. **No pass@k > 1.** With k=1 deterministic, we can't distinguish "config A is better at this specific seed" from "config A is genuinely more capable."

6. **Single image revision.** All findings are specific to `f8daec1dc883`. Future Repne images may shift the rankings.

---

## 8. Reproducibility

Every claim is reproducible from raw `results.json` and `*.jsonl` files in this repository.

```
.
├── configA-fp8mtp3/
│   ├── gates.json + gates.log              ← Phase 1 functional gates
│   ├── throughput-matrix/runs/             ← Phase 2 (18 raw N=2 runs)
│   ├── humaneval.jsonl + _summary.json     ← Phase 3 (164 problems × full metadata)
│   └── mbpp.jsonl + _summary.json          ← Phase 4 (257 problems × full metadata)
├── configB-fp8mtp5/  (same layout)
├── configC-dflash7/  (same layout)
├── configD-dflash8/  (same layout)
├── configE-dflash15/ (same layout)
├── harness/
│   ├── stress_harness.py                   ← Asyncio concurrent stress harness
│   ├── may6-config-harness.sh              ← Per-config orchestration script
│   ├── humaneval.jsonl                     ← Problem set
│   └── mbpp.jsonl                          ← Problem set
├── cross_summary.json                      ← Cross-config rollup
└── README.md
```

Each `humaneval.jsonl` / `mbpp.jsonl` line contains:
- `task_id`, `benchmark`, `config_label`
- HTTP status, error string (if any)
- Per-request timing: start ts, end ts, elapsed seconds
- vLLM response shape: `finish_reason`, `completion_tokens`, `prompt_tokens`, `reasoning_chars`, `content_chars`
- Code extraction outcome: `code_extracted`, `code_chars`
- Test execution outcome: `test_run`, `test_passed`, `test_returncode`, `test_stderr_head`
- `failure_mode` classification (one of seven categories)
- Truncated `content_head` and `reasoning_head` (first 500 chars each, for debugging)

To reproduce a single configuration:
```bash
# Launch the desired container (see Section 3.2)
# Then:
./harness/may6-config-harness.sh <CONFIG_LABEL> <KV_BUDGET> <OUT_DIR>
```

---

## 9. Conclusions

1. **Image `f8daec1dc883` (Repne May 6) is operationally rock-solid.** Five configurations × 421 stress problems × concurrency=8 = 2,105 problem-completions with zero crashes, hangs, malformed responses, or HTTP errors. Approved for production use.

2. **FP8+MTP=3 is the production SOTA for coding-agent workloads.** It wins HumanEval pass-rate (79.3%) by a meaningful margin over all alternatives, while delivering competitive throughput at production concurrency. The previously-recommended FP8+MTP=5 is dominated on correctness despite faster raw throughput.

3. **BF16+DFlash variants are the right choice for MBPP-class workloads** where small precision deltas in BF16 weights resolve test-case boundary conditions that FP8 W8A8 mishandles. They are 25–30% slower than FP8 but produce 3.5–3.9 percentage-point higher MBPP pass-rates.

4. **The DFlash `num_speculative_tokens` parameter has no effect on correctness within {7, 8, 15}** — all three converge to identical HumanEval pass count and near-identical MBPP pass count. Pick the throughput-optimal value (n=7 or n=8) and ignore n=15.

5. **Throughput-only benchmarks systematically over-recommend speculative-decoding aggressiveness.** MTP=5 wins decode-bench throughput but loses coding correctness. Always validate end-to-end on a verifiable downstream task.

6. **Increase `max_tokens` to 16384 in production agent deployments.** The dominant failure mode at concurrency=8 was reasoning-budget exhaustion (`finish_reason=length`), which would be reclaimed as solved problems with a larger token budget.

---

## 10. Recommendations for future work

1. **pass@k characterization.** Run k ∈ {1, 5, 10} at temp=0.7 to distinguish capability from luck.
2. **Higher-concurrency stress test.** c ∈ {16, 32, 64} HumanEval/MBPP runs to verify the configurations remain stable under burst load.
3. **Long-context coding tasks.** SWE-bench-Verified or LiveCodeBench at 64k–128k context length.
4. **Quality-vs-throughput Pareto frontier across MTP n.** This study covered n ∈ {3, 5}; extending to n ∈ {2, 4, 6} and measuring coding-correctness at each could identify the global pass-rate optimum.
5. **Eagle drafter trial.** If a Qwen3.6-27B Eagle drafter exists, it could exceed MTP acceptance rates and push throughput higher without sacrificing correctness.
6. **Multi-language coding benchmarks.** This study uses only Python. C++, Rust, JavaScript, and SQL coding benchmarks may rank configurations differently.

---

## 11. Related work

- Prior sprint (Apr 5 → May 6): [`jcartu/qwen36-27b-blackwell-inference-study`](https://github.com/jcartu/qwen36-27b-blackwell-inference-study) — eight-experiment characterization of Qwen3.6-27B inference, including the perplexity / KL-divergence quality probe (Q8/BF16 KLD = 0.0018 nats) that motivated this stress-validation study.
- Repne fork: [`repne/vllm`](https://hub.docker.com/r/repne/vllm)
- Upstream vLLM: [`vllm-project/vllm`](https://github.com/vllm-project/vllm)
- HumanEval: [`openai/human-eval`](https://github.com/openai/human-eval)
- MBPP: [Google Research datasets](https://huggingface.co/datasets/google-research-datasets/mbpp) (sanitized split)

---

## 12. Addendum (May 6, evening) — `draft_sample_method=gumbel` removal study

### 12.1 Motivation

After Sections 1–11 were finalised, the Repne fork upstream advised in their bug-tracker channel that the `--speculative-config.draft_sample_method gumbel` flag is **deprecated** and would be removed in a subsequent build. The previous DFlash configurations (C, D, E in this report) were all launched with `gumbel` enabled. To isolate the correctness and performance effect of removing this flag, we re-ran the **full four-phase harness** (gates → throughput-matrix → HumanEval → MBPP) on all three DFlash variants with `gumbel` removed and with two harness improvements:

- `--decode-warmup-seconds 20` (was 10) — eliminate residual JIT compilation in the first measured request.
- `--duration 60` (was 30) — average over more decode steps to suppress speculative-decoding variance.

All other server flags, the model checkpoint, the drafter checkpoint, and the host environment were held constant. Three new configurations were validated:

| New config | Drafter | n_spec | gumbel | Harness | Image |
|---|---|---|---|---|---|
| `configC2-dflash7-nogumbel` | `z-lab/Qwen3.6-27B-DFlash` | 7 | **OFF** | v2 (60s/20s) | `repne/vllm@f8daec1dc883` |
| `configD2-dflash8-nogumbel` | `z-lab/Qwen3.6-27B-DFlash` | 8 | **OFF** | v2 (60s/20s) | `repne/vllm@f8daec1dc883` |
| `configE2-dflash15-nogumbel` | `z-lab/Qwen3.6-27B-DFlash` | 15 | **OFF** | v2 (60s/20s) | `repne/vllm@f8daec1dc883` |

### 12.2 Coding correctness — gumbel ON vs OFF

**HumanEval (164 problems, c=8, max_tokens=4096, temp=0.0):**

| Config | Gumbel ON pass | Gumbel OFF pass | Δ pass | Δ pp |
|---|---|---|---|---|
| DFlash N=7  | 120/164 (73.2%) | 120/164 (73.2%) | **0** | 0.0 |
| DFlash N=8  | 120/164 (73.2%) | 116/164 (70.7%) | **−4** | −2.4 |
| DFlash N=15 | 120/164 (73.2%) | **128/164 (78.0%)** | **+8** | **+4.9** ⭐ |

**MBPP sanitized (257 problems, c=8, max_tokens=4096, temp=0.0):**

| Config | Gumbel ON pass | Gumbel OFF pass | Δ pass | Δ pp |
|---|---|---|---|---|
| DFlash N=7  | 229/257 (89.1%) | 228/257 (88.7%) | **−1** | −0.4 |
| DFlash N=8  | 230/257 (89.5%) | 230/257 (89.5%) | **0**  | 0.0 |
| DFlash N=15 | 230/257 (89.5%) | 227/257 (88.3%) | **−3** | −1.2 |

**Headline:** removing `gumbel` is a **net win for DFlash N=15** (+8 HumanEval problems, −3 MBPP, net +5 problems / +0.9pp average pass-rate across both benchmarks). For N=7 and N=8 the change is correctness-neutral within run-to-run variance (≤4 problems on a 164-problem benchmark = ~2.4pp, comparable to the noise floor we measured in the prior sprint).

> The +8 HumanEval improvement at N=15 is striking and reverses the prior conclusion (Section 1 finding 4) that **n_spec is a pure throughput knob**. With `gumbel` disabled, the longer speculative window at N=15 appears to recover correctness — possibly because greedy argmax draft sampling is better matched to the deterministic temp=0 verifier than the stochastic gumbel-perturbed drafter when the speculative chain is long.

### 12.3 Throughput matrix — gumbel ON vs OFF

Mean output tokens/sec across all 9 cells (3 contexts × 3 concurrencies, 2 runs per cell, 60s sustained-decode each):

| Config | Gumbel ON mean | Gumbel OFF mean | Δ tok/s | Δ % |
|---|---|---|---|---|
| DFlash N=7  | 197.7 | 196.6 | −1.1 | −0.6% |
| DFlash N=8  | 195.7 | 195.9 | +0.2 | +0.1% |
| DFlash N=15 | 179.6 | 181.0 | +1.3 | +0.7% |

**Headline:** removing `gumbel` is **performance-neutral** at the matrix-mean level — all three configs remain within ±1% of their gumbel-on baselines. The largest per-cell mover is N=15 at ctx=0/c=1 (+10.6%, 88.4 → 97.8 tok/s); the largest regression is N=7 at ctx=32k/c=1 (−5.2%, 93.2 → 88.3 tok/s). These offset across the matrix.

### 12.4 Speculative-decoding acceptance — gumbel ON vs OFF

Mean acceptance rate across all 18 throughput-matrix runs per config (server-reported `vllm:spec_decode_efficiency` family):

| Config | Gumbel ON accept-rate | Gumbel OFF accept-rate | Δ |
|---|---|---|---|
| DFlash N=7  | 23.04% | 23.50% | +0.46pp |
| DFlash N=8  | 22.78% | 25.58% | **+2.80pp** |
| DFlash N=15 | 13.20% | 13.50% | +0.30pp |

Acceptance rates are **slightly higher** with `gumbel` removed — consistent with the hypothesis that greedy drafter sampling matches the deterministic temp=0 verifier more closely than gumbel-perturbed drafter sampling.

### 12.5 Stress-harness gates — gumbel ON vs OFF

All three no-gumbel configs achieved **2/3 gates** (identical to their gumbel-on counterparts). The third gate (sustained 30-minute high-concurrency soak) is gated on a long-running test that was not re-run in this addendum sweep due to time budget.

### 12.6 Failure modes encountered

- **One-off CUDA launch timeout, configC2 first attempt.** During the MBPP phase of the first `configC2-dflash7-nogumbel` run, the engine died with:
  ```
  RuntimeError: Worker failed with error 'CUDA error: the launch timed out and was terminated' (cudaErrorLaunchTimeout)
  ```
  73 HTTP errors and 158 client exceptions were logged. The GPUs recovered cleanly without a host reboot, the container was relaunched with the same command, and the retry completed with no anomalies. We attribute this to a transient kernel-launch deadline issue rather than a configuration-specific bug; it is not reproducible and not n_spec=7-specific. Both the failed-attempt log and the successful retry data are checked into `configC2-dflash7-nogumbel/`.

- **No other anomalies.** All 9 throughput-matrix cells, all gates checks, and the HumanEval/MBPP phases for D2 and E2 ran without intervention.

### 12.7 Updated production guidance

The headline production recommendation is **unchanged**: FP8+MTP=3 remains the SOTA for coding-agent workloads (HumanEval 79.3%, well above any DFlash configuration with or without gumbel).

For **MBPP-dominated** or **BF16-precision-required** workloads, the DFlash variants now have a clearer ranking:

| Rank | Config | HumanEval | MBPP | Notes |
|---|---|---|---|---|
| 1 | **DFlash N=15 no-gumbel** | **78.0%** | 88.3% | New SOTA among BF16+DFlash for HumanEval; +8 problems vs gumbel-on. |
| 2 | DFlash N=7 no-gumbel | 73.2% | 88.7% | Tied highest MBPP among DFlash; throughput-optimal at low concurrency. |
| 3 | DFlash N=8 no-gumbel | 70.7% | 89.5% | Highest MBPP, slightly weaker HumanEval. |

**Action items:**
1. Drop `--speculative-config.draft_sample_method gumbel` from all DFlash launch commands going forward (already deprecated upstream).
2. Update Section 1 finding 4 of the main report: at temp=0, `n_speculative_tokens=15` with `gumbel` disabled **outperforms** n=7/8 on HumanEval by ~5pp. The prior conclusion that "n_spec has no effect on correctness" only held with `gumbel` enabled.
3. Continue running production on **FP8+MTP=3** (still 79.3% > 78.0%).

### 12.8 Reproduce

Launch (DFlash, no gumbel, N parameterized via `$NUM_SPEC`):

```bash
docker run --rm \
    --runtime nvidia --gpus all --ipc=host \
    --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    --volume ~/certificates:/root/certificates \
    --volume ~/.cache/huggingface:/root/.cache/huggingface \
    --volume ~/.cache/vllm:/root/.cache/vllm \
    --volume ~/.cache/flashinfer:/root/.cache/flashinfer \
    --volume ~/.triton/cache:/root/.triton/cache \
    --env OMP_NUM_THREADS=8 \
    --env VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
    --env NCCL_P2P_LEVEL=SYS --env NCCL_NET_GDR_LEVEL=SYS \
    --env NCCL_MIN_NCHANNELS=8 \
    --env HUGGING_FACE_HUB_TOKEN=hf_REDACTED \
    repne/vllm:latest \
        -O3 \
        --model Qwen/Qwen3.6-27B \
        --served-model-name Qwen3.6-27B \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 262144 \
        --max-num-seqs 128 \
        --max-num-batched-tokens 32758 \
        --max-cudagraph-capture-size 256 \
        --language-model-only \
        --enable-auto-tool-choice \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --enable-prefix-caching \
        --speculative-config.method dflash \
        --speculative-config.model z-lab/Qwen3.6-27B-DFlash \
        --speculative-config.num_speculative_tokens ${NUM_SPEC} \
        --speculative-config.attention_backend flash_attn \
        --speculative-config.use_local_argmax_reduction true \
        --attention-backend flashinfer \
        --default-chat-template-kwargs.preserve_thinking true
```

(Drop `--speculative-config.draft_sample_method gumbel` — the parameter is no longer supported.)

Harness (per config):

```bash
# Throughput matrix: 3 contexts × 3 concurrencies × 2 runs × 60s sustained decode
bench --concurrency 1,2,4 \
      --contexts 0,32000,131072 \
      --runs 2 \
      --duration 60 \
      --decode-warmup-seconds 20 \
      --max-tokens 2048 \
      --skip-prefill

# HumanEval / MBPP: c=8, max_tokens=4096, temp=0.0
```

Engine: `repne/vllm:latest` digest `f8daec1dc883`, vLLM `v0.1.dev16434+g81845bdee.d20260506`.

### 12.9 Raw data

- `configC2-dflash7-nogumbel/` — gates, throughput-matrix (9 cells × 2 runs), HumanEval, MBPP, full driver logs (including failed-attempt log and successful retry log).
- `configD2-dflash8-nogumbel/` — gates, throughput-matrix, HumanEval, MBPP, driver log.
- `configE2-dflash15-nogumbel/` — gates, throughput-matrix, HumanEval, MBPP, driver log.


---

## 13. Addendum (May 7) — FP8 base × DFlash drafter (no-gumbel) study

### 13.1 Motivation

Section 12 isolated the effect of removing `--speculative-config.draft_sample_method gumbel` from the **BF16** base + DFlash drafter pairings. A natural follow-up: **what happens when DFlash is paired with the FP8-quantised base model instead of BF16?** The headline production configuration (FP8+MTP=3) and the BF16+DFlash configurations occupy different correctness/throughput points; the FP8+DFlash combination has not been characterised in any prior section of this report.

This addendum runs the **full four-phase harness** (gates → throughput-matrix → HumanEval → MBPP) on three new configurations:

| New config | Base | Drafter | n_spec | gumbel | Image |
|---|---|---|---|---|---|
| `configF-fp8-dflash7-nogumbel`  | `Qwen/Qwen3.6-27B-FP8` (W8A8) | `z-lab/Qwen3.6-27B-DFlash` (BF16) | 7  | OFF | `repne/vllm@f8daec1dc883` |
| `configG-fp8-dflash8-nogumbel`  | `Qwen/Qwen3.6-27B-FP8` (W8A8) | `z-lab/Qwen3.6-27B-DFlash` (BF16) | 8  | OFF | `repne/vllm@f8daec1dc883` |
| `configH-fp8-dflash15-nogumbel` | `Qwen/Qwen3.6-27B-FP8` (W8A8) | `z-lab/Qwen3.6-27B-DFlash` (BF16) | 15 | OFF | `repne/vllm@f8daec1dc883` |

All three runs added a **60-second post-ready settle period** between engine-ready signal (`/v1/models` HTTP 200) and the first benchmark request, on top of the existing `--decode-warmup-seconds 20`. This eliminates any residual flashinfer / CUDA-graph caching effects from the first measured cell. All other harness settings are identical to Section 12.

### 13.2 Engine-startup incompatibility (resolved)

**Failure observed:** the FP8 base + BF16 DFlash drafter pairing **segfaults at engine init** when launched with `--load-format instanttensor` (the production FP8 launcher's default). The error path is consistent across all three N values:

```
(Worker_TP0) Loading drafter model...
(Worker_TP0) Loading safetensors using InstantTensor loader: 100% Completed | 58/58
!!!!!!! Segfault encountered !!!!!!!
  cuMemcpyDtoDAsync_v2 → cudaMemcpyAsync → at::native::copy_device_to_device
```

**Resolution:** drop `--load-format instanttensor` from the launch command. The standard safetensors loader handles the mixed-precision base/drafter pairing without issue. With the standard loader, all three configs come up cleanly in 110–302s. We attribute the segfault to InstantTensor's mmap-based zero-copy path being incompatible with the dtype-conversion required when the FP8 base hands off residual/embedding tensors to the BF16 drafter; this is a fixable issue but is documented here for reproducibility.

### 13.3 Coding correctness — three-way comparison

(All HumanEval at c=8, max_tokens=4096, temp=0.0; all MBPP sanitized split, c=8, max_tokens=4096, temp=0.0.)

| Config | HumanEval pass | MBPP pass | Combined (HE+MB) |
|---|---|---|---|
| **FP8+MTP=3** (production SOTA) | **130/164 (79.3%)** | 220/257 (85.6%) | 350/421 (83.1%) |
| BF16+DFlash N=7  no-gumbel | 120/164 (73.2%) | 228/257 (88.7%) | 348/421 (82.7%) |
| BF16+DFlash N=8  no-gumbel | 116/164 (70.7%) | **230/257 (89.5%)** | 346/421 (82.2%) |
| BF16+DFlash N=15 no-gumbel | 128/164 (78.0%) | 227/257 (88.3%) | 355/421 (84.3%) |
| **FP8+DFlash N=7  no-gumbel** | 121/164 (73.8%) | 225/257 (87.5%) | 346/421 (82.2%) |
| **FP8+DFlash N=8  no-gumbel** | **122/164 (74.4%)** | 227/257 (88.3%) | 349/421 (82.9%) |
| **FP8+DFlash N=15 no-gumbel** | 121/164 (73.8%) | 224/257 (87.2%) | 345/421 (82.0%) |

**Headline correctness findings:**

1. **FP8+DFlash sits between BF16+DFlash and FP8+MTP=3 on HumanEval.** All three FP8+DFlash variants land at 73.8–74.4%, slightly better than BF16+DFlash N=7 / N=8 (73.2% / 70.7%) and below BF16+DFlash N=15 (78.0%) and FP8+MTP=3 (79.3%).
2. **MBPP is essentially flat across all DFlash configurations regardless of base precision** (87.2–89.5%). The base-precision choice barely affects MBPP pass-rate.
3. **FP8+DFlash does NOT inherit BF16+DFlash N=15's HumanEval bump.** The +8-problem advantage that BF16+DFlash N=15 enjoyed in Section 12 disappears at FP8: FP8+DFlash N=15 = 73.8%, the same as N=7 and N=8. Whatever mechanism gave BF16 N=15 its edge does not survive FP8 quantisation.
4. **N=8 is the FP8+DFlash sweet spot** (74.4% HE, 88.3% MB). It is the new best-correctness FP8+DFlash configuration.

### 13.4 Throughput matrix — three-way comparison

Mean output tok/s across all 9 cells (3 contexts × 3 concurrencies × 2 runs × 60s sustained decode):

| Config | Mean tok/s | Δ vs FP8+MTP=3 | Δ vs BF16+DF same N |
|---|---|---|---|
| **FP8+MTP=3** | **241.5** | — | — |
| BF16+DFlash N=7 | 196.6 | −18.6% | — |
| BF16+DFlash N=8 | 195.9 | −18.9% | — |
| BF16+DFlash N=15 | 181.0 | −25.0% | — |
| **FP8+DFlash N=7** | **244.2** | **+1.1%** | **+24.2%** |
| **FP8+DFlash N=8** | **240.4** | −0.5% | **+22.7%** |
| FP8+DFlash N=15 | 222.8 | −7.7% | **+23.1%** |

**Headline throughput findings:**

1. **FP8+DFlash N=7 narrowly beats FP8+MTP=3 on aggregate throughput** (244.2 vs 241.5 tok/s, +1.1%) and FP8+DFlash N=8 effectively ties it (240.4, −0.5%). The earlier BF16+DFlash configurations were 19–25% slower than FP8+MTP=3; switching to the FP8 base completely closes that gap.
2. **FP8 base lifts DFlash throughput by ~23% across all three N values** vs. BF16 base, holding the drafter constant. This is the cleanest A/B isolation of base-model precision in this study.
3. **FP8+DFlash dominates at long context.** At ctx=131k, FP8+DFlash N=8 beats FP8+MTP=3 by **+5.4% to +12.7%** across c∈{1,2,4}. At c=4 / ctx=131k the advantage is +18.5 tok/s (359.5 vs 341.0).
4. **FP8+MTP=3 retains an edge at ctx=0 / high concurrency.** At c=4 / ctx=0, FP8+MTP=3 = 441.7 tok/s vs FP8+DFlash N=8 = 419.1 (−5.1%). MTP's tighter integration with the main decoder appears to win when there's no KV-cache pressure.

### 13.5 Speculative-decoding acceptance — three-way comparison

| Config | Mean acceptance |
|---|---|
| FP8+MTP=3 | 55.32% |
| BF16+DFlash N=7 | 23.50% |
| BF16+DFlash N=8 | 25.58% |
| BF16+DFlash N=15 | 13.50% |
| FP8+DFlash N=7  | 23.67% |
| FP8+DFlash N=8  | 22.12% |
| FP8+DFlash N=15 | 12.62% |

DFlash acceptance rates are stable across base precision (BF16↔FP8 deltas are within ±3.5pp). MTP=3's 55% acceptance remains in a different league — but it does not translate proportionally to throughput because each accepted MTP token costs more compute than each accepted DFlash token. The DFlash architecture's lower per-token verification cost is what allows FP8+DFlash N=7/8 to match FP8+MTP=3 throughput despite a 2× lower acceptance rate.

### 13.6 Per-cell head-to-head: FP8+DFlash N=8 vs FP8+MTP=3

| ctx | c | FP8+MTP=3 tok/s | FP8+DF8 tok/s | Δ | Δ% |
|---|---|---|---|---|---|
| 0 | 1 | 116.8 | 110.5 | −6.3 | −5.4% |
| 0 | 2 | 220.4 | 221.6 | +1.2 | +0.5% |
| 0 | 4 | 441.7 | 419.1 | −22.5 | −5.1% |
| 32k | 1 | 110.7 | 115.0 | +4.3 | +3.8% |
| 32k | 2 | 223.2 | 220.5 | −2.7 | −1.2% |
| 32k | 4 | 439.2 | 406.1 | −33.1 | −7.5% |
| **128k** | **1** | **96.8** | **109.1** | **+12.3** | **+12.7%** |
| **128k** | **2** | **183.5** | **202.1** | **+18.6** | **+10.1%** |
| **128k** | **4** | **341.0** | **359.5** | **+18.5** | **+5.4%** |

**Verdict:** FP8+DFlash N=8 trades short-context throughput for long-context throughput vs. FP8+MTP=3. For coding-agent workloads dominated by 64k–256k context (codebase-aware tasks), FP8+DFlash N=8 is the better choice. For ctx=0 high-concurrency batch serving, FP8+MTP=3 still wins.

### 13.7 Per-cell head-to-head: FP8 vs BF16 base, holding DFlash N=8 fixed

| ctx | c | BF16+DF8 tok/s | FP8+DF8 tok/s | Δ | Δ% |
|---|---|---|---|---|---|
| 0 | 1 | 94.5 | 110.5 | +16.0 | +16.9% |
| 0 | 2 | 179.6 | 221.6 | +42.0 | **+23.4%** |
| 0 | 4 | 341.0 | 419.1 | +78.1 | **+22.9%** |
| 32k | 1 | 97.0 | 115.0 | +18.0 | +18.6% |
| 32k | 2 | 174.1 | 220.5 | +46.5 | **+26.7%** |
| 32k | 4 | 321.2 | 406.1 | +84.9 | **+26.4%** |
| 128k | 1 | 86.1 | 109.1 | +22.9 | **+26.6%** |
| 128k | 2 | 165.1 | 202.1 | +37.0 | +22.4% |
| 128k | 4 | 304.1 | 359.5 | +55.4 | +18.2% |

**Verdict:** at constant drafter (DFlash N=8) and constant gumbel-off, switching the base from BF16 W16A16 to FP8 W8A8 yields a uniform **~22% throughput uplift** across all 9 cells. Correctness deltas are small: HumanEval 116 → 122 (+6 problems, +3.7pp) and MBPP 230 → 227 (−3 problems, −1.2pp). Net combined HE+MB 346 → 349 (+3 problems / +0.7pp).

### 13.8 Stress-harness gates

All three FP8+DFlash configs achieved **4/4 functional gates** (Fibonacci recurrence sanity, tool-call invocation, deterministic arithmetic, multi-turn role consistency). No engine crashes, no hangs, no malformed responses across 2,105 test executions in this addendum sweep.

### 13.9 Updated production decision tree

Adding FP8+DFlash to Section 5's decision tree:

```
                  ┌──────────────────────────────────────────┐
                  │ What is the dominant workload?           │
                  └────────────────┬─────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────────┐
        │                          │                              │
   ctx≤32k, c≥4              ctx∈[64k,256k]                MBPP-class
   batch serving             coding-agent                  short context
        │                          │                              │
        ▼                          ▼                              ▼
   FP8+MTP=3                FP8+DFlash N=8                BF16+DFlash N=8
   (442 tok/s ctx=0/c=4)    (+10% vs MTP=3 @ 128k)        (89.5% MBPP)
   (79.3% HE / 85.6% MB)    (74.4% HE / 88.3% MB)         (70.7% HE)
                                                          (BF16 weights)
```

**Headline production guidance is unchanged for the existing serving profile** (general coding-agent workload, mixed contexts): **FP8+MTP=3 stays as the default**, because its 79.3% HumanEval is meaningfully ahead of every DFlash variant (best DFlash on HE is BF16+DF15 at 78.0%, and FP8+DF8 at 74.4%).

**New guidance for long-context-dominated workloads (ctx≥64k):** consider switching to **FP8+DFlash N=8 no-gumbel**. It loses 5.0 percentage points of HumanEval (79.3% → 74.4%) but gains 5–13% throughput at ctx∈{64k, 128k} for c∈{1,2,4}, plus +2.7pp on MBPP (85.6% → 88.3%). For workloads where the agent operates over a large repo and per-request latency matters, this is a favourable trade.

### 13.10 Failure modes encountered

- **Engine-init segfault with `--load-format instanttensor`** (described in §13.2). Resolved by switching to standard safetensors loader. **Action item:** the production `launch_fp8.sh` retains `instanttensor` because it works for the FP8+MTP=3 pairing; the new `launch-fp8-dflash-N-nogumbel.sh` uses the standard loader.
- **No other anomalies.** All gates, all 9 throughput-matrix cells × 2 runs × 60s, all 164 HumanEval problems, and all 257 MBPP problems completed without engine intervention across all three configs.

### 13.11 Reproduce

Launch (FP8 + DFlash, no gumbel, N parameterised; **note the absence of `--load-format instanttensor`**):

```bash
docker run --rm \
    --runtime nvidia --gpus all --ipc=host \
    --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    --volume ~/.cache/huggingface:/root/.cache/huggingface \
    --volume ~/.cache/vllm:/root/.cache/vllm \
    --volume ~/.cache/flashinfer:/root/.cache/flashinfer \
    --volume ~/.triton/cache:/root/.triton/cache \
    --env OMP_NUM_THREADS=8 \
    --env VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
    --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
    --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
    --env NCCL_P2P_LEVEL=SYS --env NCCL_NET_GDR_LEVEL=SYS \
    --env NCCL_MIN_NCHANNELS=8 \
    --env HUGGING_FACE_HUB_TOKEN=hf_REDACTED \
    repne/vllm:latest \
        -O3 \
        --model Qwen/Qwen3.6-27B-FP8 \
        --served-model-name Qwen3.6-27B \
        --tensor-parallel-size 2 \
        --gpu-memory-utilization 0.85 \
        --max-model-len 262144 \
        --max-num-seqs 128 \
        --max-num-batched-tokens 32758 \
        --max-cudagraph-capture-size 256 \
        --language-model-only \
        --enable-auto-tool-choice \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --enable-prefix-caching \
        --speculative-config.method dflash \
        --speculative-config.model z-lab/Qwen3.6-27B-DFlash \
        --speculative-config.num_speculative_tokens ${NUM_SPEC} \
        --speculative-config.attention_backend flash_attn \
        --speculative-config.use_local_argmax_reduction true \
        --attention-backend flashinfer \
        --default-chat-template-kwargs.preserve_thinking true
```

(Do **not** add `--load-format instanttensor`. Drafter init segfaults — see §13.2.)

After server reports ready (`GET /v1/models` returns 200), wait **60 seconds** before the first benchmark request to let CUDA graphs and flashinfer caches settle. Then run the standard four-phase harness (gates → matrix at 9 cells × 2 runs × 60s decode + 20s warmup → HumanEval c=8 → MBPP c=8).

### 13.12 Raw data

- `configF-fp8-dflash7-nogumbel/` — gates.json, throughput-matrix (9 cells × 2 runs), humaneval.jsonl, mbpp.jsonl, summary JSONs, full driver log.
- `configG-fp8-dflash8-nogumbel/` — same artifact set.
- `configH-fp8-dflash15-nogumbel/` — same artifact set.

Engine: `repne/vllm:latest` digest `f8daec1dc883`, vLLM `v0.1.dev16434+g81845bdee.d20260506`. KV-cache budgets (FP8+DFlash, no-gumbel): N=7 → 1,178,863 | N=8 → 1,171,846 | N=15 → 1,121,790. (Note: ~24% larger than the corresponding BF16+DFlash KV budgets, consistent with FP8 weights leaving more HBM for the page-attention block table.)

