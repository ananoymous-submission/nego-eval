"""Negotiation tournament: Traditional vs. LLMAgent.

Pits the configured agents against each other across all combinations:
- LLMAgent vs LLMAgent
- LLMAgent vs each configured traditional agent (both sides)
- Traditional vs traditional (round-robin baseline over the configured list)
- LLMAgent vs each configured adversary (robustness sweep, both sides)

ALTERNATING protocol only. Council sessions are NOT run here — they live in
`evaluation/run_council_sessions.py` and write to `council_logs/`.

Configuration lives in tournament_config.py at the project root.
"""

import datetime
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

# Traditional agents (in main/agents/) import each other via top-level
# `agents.*` and `nenv.*` paths. Put `main/` on sys.path so those resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "main"))

from dotenv import load_dotenv

load_dotenv()

import tournament_config as cfg
from main.LLMAgent import LLMAgent
from main.adversaries import AdversarialAgent
from main.nenv.Agent import AbstractAgent
from main.nenv.Preference import Preference
from main.nenv.Session import Session


# Tournament-execution settings (not configured per-run)
LOG_DIRECTORY = "session_logs"
MAX_CONCURRENT_NEGOTIATIONS = os.cpu_count() or 1
PROTOCOL = "ALTERNATING"

# Sourced from tournament_config.py
PURE_LLMS = cfg.PURE_LLMS
DOMAINS = cfg.DOMAINS
DEADLINE_ROUND = cfg.DEADLINE_ROUND
DEADLINE_TIME = cfg.DEADLINE_TIME
TRADITIONAL_OPPONENTS = cfg.TRADITIONAL_OPPONENTS
ADVERSARIES = getattr(cfg, "ADVERSARIES", {})
ROBUSTNESS_ADVERSARY_BIDDER = getattr(cfg, "ROBUSTNESS_ADVERSARY_BIDDER", None)

# Tags for kind of side. KIND_COUNCIL stays here so side_label() returns
# "Council" for the council runner's filename construction — the council
# itself is built in evaluation/run_council_sessions.py, not here.
KIND_LLM = "LLM"
KIND_TRAD = "TRAD"
KIND_ADV = "ADV"
KIND_COUNCIL = "COUNCIL"


def domain_dir_name(domain: int) -> str:
    """Map a domain int from the config to its folder under main/domains/."""
    return f"domain{domain}"


def get_profile_path(domain: int, profile_type: str) -> str:
    return f"main/domains/{domain_dir_name(domain)}/profile{profile_type}.json"


def make_agent(kind: str, identifier: str, preference: Preference) -> AbstractAgent:
    """Build an agent given its kind and identifier.

    kind="LLM":     identifier is the model_name.
    kind="TRAD":    identifier is the display name from TRADITIONAL_OPPONENTS.
    kind="ADV":     identifier is the display name from cfg.ADVERSARIES; the
                    adversary uses cfg.ROBUSTNESS_ADVERSARY_BIDDER for its
                    bids and cfg.PURE_LLMS[0] as the persona-dialogue model.
    """
    if kind == KIND_LLM:
        return LLMAgent(
            preference=preference,
            estimators=[],
            model_name=identifier,
            protocol=PROTOCOL,
        )
    if kind == KIND_TRAD:
        cls = TRADITIONAL_OPPONENTS[identifier]
        agent = cls(preference, 180, [])
        # Session always calls act(t=..., chat_history=...). Traditional agents
        # define act(self, t) only — wrap so they ignore the extra kwarg.
        original_act = agent.act
        agent.act = lambda t, **_kwargs: original_act(t)
        return agent
    if kind == KIND_ADV:
        if ROBUSTNESS_ADVERSARY_BIDDER is None:
            raise ValueError("ADV requires cfg.ROBUSTNESS_ADVERSARY_BIDDER to be set.")
        if not PURE_LLMS:
            raise ValueError("ADV needs at least one model in cfg.PURE_LLMS for its persona dialogue.")
        persona = ADVERSARIES[identifier]
        return AdversarialAgent(
            preference=preference,
            session_time=180,
            estimators=[],
            persona=persona,
            bidder_cls=ROBUSTNESS_ADVERSARY_BIDDER,
            model_name=PURE_LLMS[0],
        )
    raise ValueError(f"Unknown agent kind: {kind!r}")


def side_label(kind: str, identifier: str) -> str:
    """File-safe label for log paths."""
    if kind == KIND_ADV:
        # Adversary identifiers (e.g. "Adversary-threat") already carry a
        # readable name; emit them as-is without the kind prefix.
        return identifier.replace("/", "_").replace(":", "_")
    if kind == KIND_COUNCIL:
        # Identifier is the domain int — the council is a single artifact
        # regardless of which side it plays, so collapse to a bare label.
        return "Council"
    safe = identifier.replace("/", "_").replace(":", "_")
    return f"{kind}-{safe}"


def session_log_path(side_a: Tuple[str, str], side_b: Tuple[str, str], domain: int) -> str:
    a_label = side_label(*side_a)
    b_label = side_label(*side_b)
    return f"{LOG_DIRECTORY}/{domain_dir_name(domain)}/{a_label}_vs_{b_label}.xlsx"


def write_llm_reasoning_log(agent: AbstractAgent, log_path_xlsx: str) -> None:
    if not isinstance(agent, LLMAgent):
        return
    sidecar = log_path_xlsx.rsplit(".", 1)[0] + ".bidding.json"
    payload = {
        "agent_name": agent.name,
        "model_name": agent.model_name,
        "bids": [
            {"t": t, "reasoning": reasoning}
            for (t, reasoning) in agent.state.bid_reasoning_log
        ],
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_negotiation(
    side_a: Tuple[str, str],
    side_b: Tuple[str, str],
    domain: int,
) -> None:
    kind_a, id_a = side_a
    kind_b, id_b = side_b

    a_label = side_label(kind_a, id_a)
    b_label = side_label(kind_b, id_b)

    pref_a = Preference(get_profile_path(domain, "A"))
    pref_b = Preference(get_profile_path(domain, "B"))

    agent_a = make_agent(kind_a, id_a, pref_a)
    agent_b = make_agent(kind_b, id_b, pref_b)

    domain_dir = f"{LOG_DIRECTORY}/{domain_dir_name(domain)}"
    os.makedirs(domain_dir, exist_ok=True)
    log_path = f"{domain_dir}/{a_label}_vs_{b_label}.xlsx"

    session = Session(
        agentA=agent_a,
        agentB=agent_b,
        path=log_path,
        deadline_time=DEADLINE_TIME,
        deadline_round=DEADLINE_ROUND,
        loggers=[],
    )

    try:
        session.start()
    except Exception:
        # LLM error / agent crash mid-session. Drop any partial artifacts so
        # the run can be retried cleanly on the next tournament invocation.
        _cleanup_session_artifacts(log_path)
        raise

    write_llm_reasoning_log(agent_a, log_path)
    write_llm_reasoning_log(agent_b, log_path)


def _cleanup_session_artifacts(log_path_xlsx: str) -> None:
    """Remove the xlsx and any sidecar JSONs produced by an aborted session."""
    base = log_path_xlsx.rsplit(".", 1)[0]
    for path in (log_path_xlsx, f"{base}.bidding.json"):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def cleanup_invalid_sessions() -> int:
    """Walk every session log and delete any that ended in disagreement BEFORE
    the deadline. A session is only allowed to fail if it ran the full
    DEADLINE_ROUND-1 rounds; an earlier non-Accept terminal row means the
    session crashed (LLM error, parse failure, etc.) and the partial xlsx is
    not real data. Returns the number of sessions deleted."""
    import pandas as pd

    deadline_last_round = DEADLINE_ROUND - 1
    deleted = 0
    for xlsx in Path(LOG_DIRECTORY).rglob("*.xlsx"):
        try:
            df = pd.read_excel(xlsx, sheet_name="Session")
        except Exception:
            continue
        if df.empty:
            continue
        last = df.iloc[-1]
        if str(last["Action"]) != "Accept" and int(last["Round"]) < deadline_last_round:
            _cleanup_session_artifacts(str(xlsx))
            deleted += 1
    return deleted


def build_tasks():
    tasks = []

    for domain in DOMAINS:
        # 1. LLM × LLM (every PURE_LLM pair, both directions)
        for m_a in PURE_LLMS:
            for m_b in PURE_LLMS:
                tasks.append(((KIND_LLM, m_a), (KIND_LLM, m_b), domain))

        # 2. LLM × Traditional (both directions)
        for m in PURE_LLMS:
            for trad_name in TRADITIONAL_OPPONENTS:
                tasks.append(((KIND_LLM, m), (KIND_TRAD, trad_name), domain))
                tasks.append(((KIND_TRAD, trad_name), (KIND_LLM, m), domain))

        # 3. Traditional × Traditional (round-robin baseline)
        for a in TRADITIONAL_OPPONENTS:
            for b in TRADITIONAL_OPPONENTS:
                tasks.append(((KIND_TRAD, a), (KIND_TRAD, b), domain))

        # 4. LLM × Adversary (robustness sweep, both directions)
        for m in PURE_LLMS:
            for adv_name in ADVERSARIES:
                tasks.append(((KIND_LLM, m), (KIND_ADV, adv_name), domain))
                tasks.append(((KIND_ADV, adv_name), (KIND_LLM, m), domain))

    return tasks


def main():
    os.makedirs(LOG_DIRECTORY, exist_ok=True)

    # Stamp the LangSmith project name once in the parent so all workers
    # spawned below inherit it (no per-process timestamp drift).
    if "LANGSMITH_PROJECT_STAMPED" not in os.environ:
        os.environ["LANGSMITH_PROJECT"] = (
            os.getenv("LANGSMITH_PROJECT", "tournament")
            + "_"
            + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        os.environ["LANGSMITH_PROJECT_STAMPED"] = "1"

    # BayesianOpponentModel (used by NiceTitForTat and others) reads
    # DEADLINE_ROUND from the environment. Set it from the config so workers
    # inherit it on spawn.
    os.environ["DEADLINE_ROUND"] = str(cfg.DEADLINE_ROUND)

    # Pre-cleanup: sweep any pre-deadline disagreement sessions left from a
    # previous crashed run, so the resume-by-skip logic re-queues them.
    pre_swept = cleanup_invalid_sessions()
    if pre_swept:
        print(f"Pre-cleanup removed {pre_swept} dirty session(s) from previous run(s).")

    all_tasks = build_tasks()
    # Skip any (side_a, side_b, domain) whose log file already exists — lets
    # us resume a partial run without re-running completed sessions.
    tasks = [t for t in all_tasks if not os.path.exists(session_log_path(*t))]
    skipped = len(all_tasks) - len(tasks)
    if skipped:
        print(f"Skipping {skipped} session(s) with existing logs; running {len(tasks)} new.")
    total = len(tasks)
    if total == 0:
        return

    # Use spawn so each worker re-imports cleanly with our sys.path setup.
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_NEGOTIATIONS, mp_context=ctx) as executor:
        futures = [
            executor.submit(run_negotiation, side_a, side_b, domain)
            for (side_a, side_b, domain) in tasks
        ]
        failures = 0
        with tqdm(total=total, desc="Tournament", unit="session") as bar:
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    # Worker has already cleaned up its xlsx + sidecars; just
                    # log and keep the bar moving.
                    failures += 1
                    tqdm.write(f"[error] {type(e).__name__}: {e}")
                bar.update(1)
        if failures:
            print(f"\n{failures} session(s) errored and were not saved.")

    # Post-cleanup: any session that ended in disagreement before the deadline
    # is treated as a crash artefact (worker may not have raised). Sweep them.
    swept = cleanup_invalid_sessions()
    if swept:
        print(f"Post-cleanup removed {swept} pre-deadline disagreement session(s).")


if __name__ == "__main__":
    main()
