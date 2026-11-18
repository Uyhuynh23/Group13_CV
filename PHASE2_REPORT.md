# Phase 2 Report: Controlled 2 × 2 Evaluation of Patch-Preserving MLP vs. Query-Resampling Connectors (Flickr8k)

**Project:** A Controlled Evaluation of Patch-Preserving and Query-Resampling Connectors for Resource-Constrained Image Captioning (Group 13)  
**Repository:** `Uyhuynh23/Group13_CV` (branch `main`, commit checkpoint after `a9236de`)  
**Google Drive Data & Artifacts:** [`Group13_CV_Data`](https://drive.google.com/drive/folders/1m-90MnuFd0K8_rZR3Xj1dX2rRJRMz4mn)  
**Report Date:** 2026-11-18  

---

## 1. Phase 2 Scope & Checkpoint Status

Phase 2 completed all **12 main 2 × 2 experimental runs** (4 conditions × 3 random seeds: `42`, `123`, `999`) plus the **global-feature linear sanity baseline** (seed `42`), along with deterministic test-set evaluation under both **beam search (`beam_size=5`)** and **greedy decoding (`beam_size=1`)** on all **1,000** Flickr8k test images.

### Experimental Matrix
All conditions share a frozen **CLIP ViT-B/32** visual encoder (`49 × 768` patch features, FP16 cached) and **GPT-2 Small** decoder (`124,439,808` backbone parameters), trained with AMP, AdamW (`lr=2e-4`), and validation CIDEr early stopping:

| Condition ID | Connector | GPT-2 Adaptation | Prefix Length | Connector Params | GPT-2 Trainable Params | Total Trainable Params |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `baseline_global_linear` | Global Linear (sanity) | Frozen | 1 | 393,984 | 0 | 393,984 |
| `condA_patch_mlp_frozen` | 2-layer Patch MLP | Frozen | 49 | 1,181,184 | 0 | 1,181,184 |
| `condB_query_resampler_frozen` | Query Resampler ($K=16$) | Frozen | 16 | 992,768 | 0 | 992,768 |
| `condC_patch_mlp_lora` | 2-layer Patch MLP | LoRA (`c_attn`, $r=8, \alpha=16$) | 49 | 1,181,184 | 294,912 | 1,476,096 |
| `condD_query_resampler_lora` | Query Resampler ($K=16$) | LoRA (`c_attn`, $r=8, \alpha=16$) | 16 | 992,768 | 294,912 | 1,287,680 |

### Artifacts Verified on Google Drive
- `results/` — **25** evaluation JSON files (`1` baseline `beam5` + `24` Condition A–D `beam5`/`greedy` across seeds `42`, `123`, `999`; `N = 1,000` test images each).
- `checkpoints/` — **39** files (`13` runs × `*_best.pt`, `*_latest.pt`, `*_history.json`).

---

## 2. Aggregated Test-Set Caption Quality (Mean ± Std across 3 Seeds)

| Condition | Decode | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | CIDEr |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_global_linear` *(n=1)* | `beam5` | 40.67 | 18.98 | 8.16 | 4.38 | 29.15 | 4.69 |
| `condA_patch_mlp_frozen` | `beam5` | 43.94 ± 0.48 | 20.58 ± 0.12 | 9.07 ± 0.11 | 4.84 ± 0.09 | 30.05 ± 0.32 | 4.94 ± 0.22 |
| `condA_patch_mlp_frozen` | `greedy` | 44.42 ± 1.29 | 20.80 ± 0.57 | 9.04 ± 0.15 | 4.88 ± 0.08 | 29.87 ± 0.79 | 4.62 ± 0.33 |
| `condB_query_resampler_frozen` | `beam5` | **45.58 ± 0.85** | 22.26 ± 0.40 | 10.08 ± 0.66 | 5.55 ± 0.66 | **30.85 ± 0.25** | 5.66 ± 0.19 |
| `condB_query_resampler_frozen` | `greedy` | 45.14 ± 0.56 | **22.37 ± 0.58** | 10.27 ± 0.30 | 5.65 ± 0.23 | 30.64 ± 0.21 | 5.44 ± 0.09 |
| `condC_patch_mlp_lora` | `beam5` | 43.41 ± 0.88 | 21.06 ± 1.03 | 10.07 ± 0.79 | 5.60 ± 0.48 | 30.31 ± 0.35 | 5.74 ± 0.51 |
| `condC_patch_mlp_lora` | `greedy` | 43.12 ± 1.04 | 21.31 ± 1.02 | 10.20 ± 0.82 | 5.79 ± 0.59 | 29.95 ± 0.48 | 5.55 ± 0.49 |
| `condD_query_resampler_lora` | `beam5` | 44.52 ± 0.68 | 21.69 ± 0.17 | **10.45 ± 0.27** | **5.91 ± 0.18** | 30.27 ± 0.21 | **5.91 ± 0.09** |
| `condD_query_resampler_lora` | `greedy` | 44.35 ± 0.90 | 21.66 ± 0.58 | 10.27 ± 0.57 | 5.69 ± 0.37 | 29.87 ± 0.20 | 5.54 ± 0.29 |

---

## 3. Individual Run Breakdown (`beam5` & `greedy`)

| Condition | Seed | Decode | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr | Latency (ms/img) | Tokens/sec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_global_linear` | 42 | `beam5` | 40.67 | 4.38 | 29.15 | 4.69 | 5087.2* | 2.51* |
| `condA_patch_mlp_frozen` | 42 | `beam5` | 43.63 | 4.74 | 30.04 | 4.76 | 379.6 | 30.63 |
| `condA_patch_mlp_frozen` | 123 | `beam5` | 44.49 | 4.88 | 30.37 | 5.19 | 364.5 | 31.93 |
| `condA_patch_mlp_frozen` | 999 | `beam5` | 43.69 | 4.90 | 29.74 | 4.88 | 186.5 | 59.31 |
| `condA_patch_mlp_frozen` | 42 | `greedy` | 43.80 | 4.86 | 29.21 | 4.40 | 145.5 | 70.52 |
| `condA_patch_mlp_frozen` | 123 | `greedy` | 45.91 | 4.81 | 30.75 | 5.00 | 139.6 | 73.25 |
| `condA_patch_mlp_frozen` | 999 | `greedy` | 43.57 | 4.96 | 29.66 | 4.46 | 125.1 | 83.42 |
| `condB_query_resampler_frozen` | 42 | `beam5` | 46.56 | 5.82 | 31.03 | 5.86 | 204.7 | 53.44 |
| `condB_query_resampler_frozen` | 123 | `beam5` | 45.03 | 4.80 | 30.57 | 5.49 | 214.1 | 53.03 |
| `condB_query_resampler_frozen` | 999 | `beam5` | 45.16 | 6.03 | 30.96 | 5.63 | 183.5 | 65.25 |
| `condB_query_resampler_frozen` | 42 | `greedy` | 45.76 | 5.79 | 30.57 | 5.40 | 133.2 | 75.64 |
| `condB_query_resampler_frozen` | 123 | `greedy` | 44.67 | 5.38 | 30.48 | 5.38 | 148.8 | 72.85 |
| `condB_query_resampler_frozen` | 999 | `greedy` | 44.99 | 5.77 | 30.87 | 5.55 | 129.6 | 85.26 |
| `condC_patch_mlp_lora` | 42 | `beam5` | 42.44 | 5.08 | 29.94 | 5.20 | 419.4 | 28.53 |
| `condC_patch_mlp_lora` | 123 | `beam5` | 44.16 | 6.04 | 30.63 | 6.20 | 406.0 | 29.30 |
| `condC_patch_mlp_lora` | 999 | `beam5` | 43.62 | 5.69 | 30.35 | 5.83 | 236.4 | 50.72 |
| `condC_patch_mlp_lora` | 42 | `greedy` | 42.23 | 5.11 | 29.41 | 4.98 | 181.6 | 62.29 |
| `condC_patch_mlp_lora` | 123 | `greedy` | 44.26 | 6.04 | 30.34 | 5.81 | 193.4 | 59.16 |
| `condC_patch_mlp_lora` | 999 | `greedy` | 42.88 | 6.21 | 30.10 | 5.86 | 167.2 | 70.67 |
| `condD_query_resampler_lora` | 42 | `beam5` | 44.79 | 5.89 | 30.28 | 6.01 | 233.0 | 48.06 |
| `condD_query_resampler_lora` | 123 | `beam5` | 43.75 | 6.10 | 30.06 | 5.86 | 8889.8* | 1.30* |
| `condD_query_resampler_lora` | 999 | `beam5` | 45.03 | 5.75 | 30.47 | 5.87 | 208.2 | 55.14 |
| `condD_query_resampler_lora` | 42 | `greedy` | 44.50 | 5.77 | 30.05 | 5.72 | 176.2 | 63.25 |
| `condD_query_resampler_lora` | 123 | `greedy` | 43.39 | 5.29 | 29.65 | 5.20 | 2899.7* | 3.77* |
| `condD_query_resampler_lora` | 999 | `greedy` | 45.17 | 6.01 | 29.90 | 5.71 | 148.5 | 70.99 |

*\*Note: `baseline_global_linear` (seed 42) and `condD_query_resampler_lora` (seed 123) show ~15–40× slower evaluation latency, indicating CPU or throttled runtime execution during evaluation; these two runs should be re-benchmarked on a clean T4 GPU runtime in Phase 3 before reporting final latency comparisons.*

---

## 4. Training Efficiency & Convergence (`*_history.json`)

| Run Name | Epochs Run | Best Epoch | Best Val CIDEr | Avg Epoch Time (s) | Peak GPU VRAM (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `baseline_global_linear_seed42` | 7 | 4 | 4.99 | 95.0 | 4,018.5 |
| `condA_patch_mlp_frozen_seed42` | 5 | 2 | 4.80 | 200.4 | 6,427.5 |
| `condA_patch_mlp_frozen_seed123` | 4 | 1 | 5.02 | 199.4 | 6,427.5 |
| `condA_patch_mlp_frozen_seed999` | 6 | 3 | 4.78 | 132.4 | 6,427.5 |
| `condB_query_resampler_frozen_seed42` | 4 | 1 | 5.69 | 128.2 | 4,809.7 |
| `condB_query_resampler_frozen_seed123` | 5 | 2 | 5.91 | 128.7 | 4,809.7 |
| `condB_query_resampler_frozen_seed999` | 4 | 1 | 5.23 | 85.0 | 4,809.7 |
| `condC_patch_mlp_lora_seed42` | 6 | 3 | 5.15 | 216.5 | 6,703.1 |
| `condC_patch_mlp_lora_seed123` | 4 | 1 | 5.83 | 217.5 | 6,703.1 |
| `condC_patch_mlp_lora_seed999` | 4 | 1 | 5.59 | 142.2 | 6,703.1 |
| `condD_query_resampler_lora_seed42` | 4 | 1 | 5.95 | 132.2 | 4,981.3 |
| `condD_query_resampler_lora_seed123` | 5 | 2 | 5.45 | 140.9 | 4,981.3 |
| `condD_query_resampler_lora_seed999` | 4 | 1 | 5.48 | 89.5 | 4,981.3 |

---

## 5. Key Phase 2 Findings

1. **All 4 experimental conditions outperform the global-linear baseline** across all captioning metrics.
2. **Query Resampler outperforms Patch MLP with fewer resources:** Compressing `49 → 16` visual tokens reduces connector parameters (`992.8K` vs `1,181.2K`), cuts peak training VRAM by **~25%** (`4.81 GB` vs `6.43 GB` frozen; `4.98 GB` vs `6.70 GB` LoRA), speeds up training epochs by **~35%**, and yields higher CIDEr and BLEU-4 in both frozen (`condB` vs `condA`) and LoRA (`condD` vs `condC`) regimes.
3. **LoRA boosts CIDEr and BLEU-4:** Adding `+294,912` trainable LoRA parameters to GPT-2 `c_attn` improves `beam5` CIDEr (`condA → condC`: `4.94 → 5.74`; `condB → condD`: `5.66 → 5.91`) and `beam5` BLEU-4 (`4.84 → 5.60`; `5.55 → 5.91`), with `condD_query_resampler_lora` achieving the best overall `beam5` **CIDEr (`5.91 ± 0.09`)** and **BLEU-4 (`5.91 ± 0.18`)**. Meanwhile, `condB_query_resampler_frozen` retains the highest **BLEU-1 (`45.58 ± 0.85`)** and **ROUGE-L (`30.85 ± 0.25`)**.
4. **Early convergence:** Most runs peak at Epoch 1–3 on validation CIDEr before early stopping triggers.

---

## 6. Next Steps: Moving to Phase 3 (Analysis, Audits & Final Paper)

Per the project proposal (`proposal.md`), Phase 3 covers the evaluation and interpretability analyses using the saved Phase 2 checkpoints:

1. **Complete Metric Panel & Statistical Tests (Section V.A & VI):**
   - Compute **METEOR**, **SPICE**, and **CLIPScore (`ViT-B/16`)** across all 12 main runs + baseline.
   - Run **paired bootstrap resampling** (95% confidence intervals) over the 1,000 test images for principal pairwise comparisons (`condB vs condA`, `condD vs condC`, `condC vs condA`, `condD vs condB`) and 2 × 2 interaction analysis (RQ3).
2. **ROME & Blinded Human Hallucination Audit (Section V.B):**
   - Implement and run the **Reference-Grounded Object-Mention Error (ROME)** extractor ($ROME_i = H/O$, $ROME_s = C_H/C$) against the union of 5 reference captions.
   - Perform blinded human verification on a preregistered **100-image** subset (2 annotators + 1 adjudicator; report Cohen's $\kappa$, raw agreement, and adjudicated hallucination rates).
3. **Complex-Scene Error Analysis (Section V.C):**
   - Deterministically select the **100 test images** with the largest number of distinct concrete object nouns across their 5 references, and categorize omissions/errors (objects, attributes, actions, counts, spatial relations).
4. **Attention Visualization & Patch-Occlusion Fidelity (Section V.D):**
   - Map decoder-to-prefix attention (Patch MLP) and composed token-to-query × query-to-patch attention (Query Resampler) onto the `7 × 7` grid.
   - Run **patch-occlusion fidelity** tests ($\Delta \log p(w)$ comparing top-ranked vs. low-ranked vs. random masked patches + Spearman rank correlation).
5. **Clean T4 Latency Re-benchmark (Section V.E):**
   - Re-run evaluation timing on a clean Tesla T4 GPU for `baseline_global_linear` and `condD_query_resampler_lora` (seed `123`) so all latency numbers are measured under identical hardware conditions.
6. **Final IEEE Paper & Code Release (Section XII):**
   - Compile the IEEE conference LaTeX paper + PDF and finalize the public GitHub release.
