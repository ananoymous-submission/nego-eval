<div align="center">

# Are Large Reasoning Models Capable Negotiators?

*A unified framework for evaluating frontier LRMs against classical baselines, adversarial dialogue, and human counterparts.*


[![SIGIR '26](https://img.shields.io/badge/EMNLP-2026-blue)](https://2026.emnlp.org)

</div>

---

## TL;DR

Prior work concluded that LLMs are weak negotiators. We revisit that claim with three state-of-the-art **Large Reasoning Models** (Claude Opus 4.7, GPT-5.5, Gemini Pro 3.1) and a non-reasoning baseline (GPT-4o). Across **1,352 tournament sessions** against seven classical strategies, **384 adversarial sessions** against six manipulation personas, and a **within-subjects user study with 108 humans**, reasoning LLMs significantly outperform every classical baseline in self-utility, remain robust to adversarial dialogue, and retain their edge against human counterparts.

## 🎯 Key Results

### Tournament — top of the leaderboard (paper Table 3)

| Agent | Self-Utility | Nash Distance | Acceptance |
|---|---:|---:|---:|
| **Gemini Pro 3.1** *(LRM)* | **0.71 ± 0.20** | 0.19 | 0.96 |
| **Opus 4.7** *(LRM)* | **0.66 ± 0.20** | 0.20 | 0.98 |
| NiceTitForTat | 0.64 ± 0.15 | **0.18** | **1.00** |
| Boulware | 0.64 ± 0.22 | 0.24 | 0.92 |
| **GPT-5.5** *(LRM)* | **0.64 ± 0.20** | 0.21 | **1.00** |
| … | … | … | … |
| **GPT-4o** *(non-reasoning)* | 0.52 ± 0.41 | 0.53 | 0.65 |
| HardHeaded | 0.52 ± 0.47 | 0.68 | 0.58 |

**Three of the top five agents are LRMs; the non-reasoning baseline sits at the bottom on every metric.** 

### Adversarial robustness — acceptance under six attack personas (paper Table 4)

| Persona | GPT-4o | GPT-5.5 | Gemini Pro 3.1 | Opus 4.7 |
|---|---:|---:|---:|---:|
| No Adversary | 0.69 | 1.00 | 0.94 | 1.00 |
| Coercive Threat | **0.88 ↑** | 1.00 | 1.00 | 1.00 |
| Fabricated Constraints | **0.94 ↑** | 1.00 | 1.00 | 1.00 |
| Prompt Injection | **0.87 ↑** | 1.00 | 1.00 | 1.00 |

**Reasoning models hold near-perfect acceptance under every attack; the non-reasoning baseline is pressured into accepting offers it would otherwise refuse.**

### Human-agent study — 108 participants (paper Table 6)

| Metric | Hybrid *(classical)* | Gemini *(LRM)* |
|---|---:|---:|
| Self Utility | 0.80 ± 0.21 | **0.83 ± 0.17** † |
| Opponent Utility | 0.40 ± 0.14 | 0.43 ± 0.14 |
| Round | 6.28 ± 4.46 | **5.43 ± 3.47** † |
| Acceptance Rate | 0.95 | **0.98** |

**<sub>† statistically significant (p < 0.01).</sub>**


## 🛠️ Installation

```bash
git clone https://github.com/ananoymous-submission/nego-eval && cd nego-eval
uv sync
cp .env.example .env   # then fill in your API keys
```


## 🚀 Quick Start

Run a single LLM-vs-baseline session to verify your install (≈ 1 minute, one API call):

```python
# tournament_config.py — narrow the matrix for a smoke test
PURE_LLMS = ["openrouter/~google/gemini-pro-latest"]
TRADITIONAL_OPPONENTS = {"Boulware": BoulwareAgent}
DOMAINS = [5]
```

```bash
uv run python tournament.py
```

Outputs land in `agent-agent-sessions/domain5/` as one `.xlsx` per session.

## 🧪 Reproducing the Paper

> [!WARNING]
> A full tournament hits paid LLM APIs (~1,352 sessions for the agent-agent tournament, ~384 for the adversarial attacks). **Budget accordingly.**

### Agent-Agent Tournament

```bash
uv run python tournament.py
```

Outputs land in [`agent-agent-sessions/`](agent-agent-sessions/). All headline tournament numbers and figures (paper Tables 3, Figures 5–7) are reproduced from the notebook at [`evaluation/tournamanet_results.ipynb`](evaluation/tournamanet_results.ipynb).

### Adversarial Robustness

```bash
uv run python tournament.py
```

With `ADVERSARIES` populated in [`tournament_config.py`](tournament_config.py), the same runner produces the adversarial sessions analyzed in paper Tables 4–5.

### Human-Agent Study

> [!IMPORTANT]
> The user study was approved by our institution's ethics review board. Participants gave informed consent and were compensated at Prolific's suggested hourly rate. Released session data is anonymized as `user-1` … `user-108`.

A Gradio app conducts the within-subjects study on the Holiday domain (paper §6, Appendix D).

```bash
uv run python -m gradio user_experiment/frontend/app.py
```

The app handles username intake → surveys → preference elicitation (Fig 14) → negotiation chat (Fig 18). Per-session transcripts land in [`agent-human-sessions/`](agent-human-sessions/) as `.xlsx` files. The released dataset (216 files = 108 users × 2 protocols) is already shipped in the repo. All headline user-study numbers and figures (paper Table 6, Figure 2) are reproduced from [`evaluation/emperical/main.py`](evaluation/emperical/main.py):

```bash
uv run python evaluation/emperical/main.py
```



## 📁 Repository Structure

```
nego-eval/
├── tournament.py                # Entry point for tournament + adversarial sweeps
├── tournament_config.py         # Single source of truth: models, baselines, domains, adversaries
├── main/
│   ├── nenv/                    # NegoLog environment (Session, Bid, Preference, Agent API)
│   ├── LLMAgent/                # LLM-as-bidder agent
│   ├── HybridAgent/             # Heuristic bidder + shared LLM dialogue (user-study baseline)
│   ├── adversaries/             # 6 adversarial personas + wrapper bidder 
│   ├── agents/                  # Classical baselines (Boulware, Conceder, NTfT, SAGA, …)
│   ├── llm_components/          # Shared DSPy signatures (bidding, dialogue generation)
│   ├── heuristic_strategies/    # HybridAgent for the user study
│   └── domains/                 # NegoLog domain pack (8 used in paper; see Appendix B Table 8)
├── user_experiment/
│   ├── frontend/                # Gradio app (multi-user negotiation chat)
│   └── backend/                 # Database, negotiation service, session storage
├── evaluation/
│   ├── tournamanet_results.ipynb  # Agent-agent analysis (Table 3, Figs 5–7)
│   ├── emperical/main.py          # Human-agent analysis pipeline (Table 6, Fig 2)
│   ├── visualise.py               # Visualizations
│   └── significance_tests.py      # Wilcoxon / paired-t with effect sizes
├── agent-agent-sessions/        # Tournament session logs (8 domains)
├── agent-human-sessions/        # User-study session logs (user-1 … user-108, 216 files)
└── assets/                      # Figures
```


> [!NOTE]
> The `PURE_LLMS` list in `tournament_config.py` may diverge from the paper's snapshot as model versions are deprecated. The paper used the snapshots in Appendix A Table 7 (Opus 4.7 / Apr 16 2026, Gemini Pro 3.1 / Apr 27 2026, GPT-5.5 / Apr 24 2026, GPT-4o / Nov 20 2024) at temperature 1.0, max-tokens 32k, highest reasoning effort.

## 📝 Citation

```bibtex
@inproceedings{anonymous2026lrm,
  title     = {Are Large Reasoning Models Capable Negotiators?},
  author    = {Anonymous},
  booktitle = {Under Review},
  year      = {2026}
}
```

<!-- TODO (camera-ready): commit a LICENSE file (e.g., MIT or Apache-2.0). -->

## 🙏 Acknowledgments

This framework extends [NegoLog](https://github.com/tdgunes/NegoLog) (Doğru et al., IJCAI 2024) with LLM agents, natural-language dialogue, and adversarial personas. 
