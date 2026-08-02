"""Persona A/B harness: fixed scenario scripts x persona variants x candidate models.

Replays each scenario multi-turn against the cluster LLM via /v1/responses (thinking
suppressed, same call shape the patched s2s uses) and records every reply. Results are
cached per cell as results_cache/<model-alias>__<persona>__<scenario>.json (gitignored via
experiments/**/results_*), so re-runs only hit the API for missing cells; delete a cell
file (or the folder) to regenerate.

Personas live in personas/*.txt — edit freely, filename (minus .txt) is the label.
Scenarios live in scenarios/scenarios.json — 'user' turns are spoken student input
(several verbatim from archived session transcripts), 'intervention' turns are the app's
real system-injected prompts.

Run from this folder, with the tunnel up (localhost:8080 -> cluster Ollama):

    python3 run_harness.py                # run/refresh the whole grid
    python3 run_harness.py --report-only  # just rebuild the report from cache

Output: results_report.md (transcripts per cell + summary table).
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_URL = "http://localhost:8080/v1/responses"
MODELS = {
    "e4b": ("hf.co/ggml-org/gemma-4-E4B-it-GGUF:Q8_0", 25.4),
    "12b": ("hf.co/google/gemma-4-12b-it-qat-q4_0-gguf:Q4_0", 18.5),
    # qwen14b eliminated 27.07 (9.9 t/s, hands over solutions on ask #2); cached cells remain readable.
}
SPOKEN_WORDS_PER_SECOND = 2.5  # TTS pace proxy for "how long would this take to hear"


def responses_call(model: str, instructions: str, history: list[dict]) -> dict:
    """One /v1/responses call with thinking suppressed; history as message items."""
    payload = {
        "model": model,
        "instructions": instructions,
        "reasoning": {"effort": "none"},
        "input": [
            {
                "type": "message",
                "role": item["role"],
                "content": [{"type": "input_text" if item["role"] == "user" else "output_text", "text": item["text"]}],
            }
            for item in history
        ],
    }
    request = urllib.request.Request(
        BASE_URL, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def reply_text(api_response: dict) -> str:
    """Extract the assistant message text (skipping any reasoning items)."""
    for item in api_response.get("output", []):
        if item.get("type") == "message":
            return "".join(part.get("text", "") for part in item.get("content", []))
    return ""


def run_cell(model: str, instructions: str, turns: list[dict]) -> list[dict]:
    """Replay one scenario multi-turn; returns [{turn, role, text, reply, output_tokens, seconds}]."""
    history: list[dict] = []
    records = []
    for i, turn in enumerate(turns):
        # Interventions are injected as user-role items, matching the app's queue mechanics.
        history.append({"role": "user", "text": turn["text"]})
        started = time.monotonic()
        api_response = responses_call(model, instructions, history)
        elapsed = time.monotonic() - started
        text = reply_text(api_response)
        history.append({"role": "assistant", "text": text})
        records.append(
            {
                "turn": i,
                "role": turn["role"],
                "text": turn["text"],
                "reply": text,
                "output_tokens": api_response.get("usage", {}).get("output_tokens"),
                "seconds": round(elapsed, 2),
            }
        )
    return records


def tool_syntax_leak(reply: str) -> bool:
    """Flag tool-call syntax or thinking markup leaking into spoken text (gemma + qwen dialects).

    Covers brace style (12B native markup), paren style (E4B pseudo-calls), bare tool-name
    mentions at reply start, and code fences (unspeakable in TTS).
    """
    markers = ("<|\"|>", "<|channel", "<|tool_call", "<think>", "```")
    tool_names = ("play_emotion", "move_head", "sweep_look", "dance")
    if any(marker in reply for marker in markers):
        return True
    return any(f"{name}(" in reply or f"{name}{{" in reply or f"{name}:" in reply for name in tool_names)


def build_report(cells: dict) -> str:
    lines = ["# Persona harness report\n"]
    lines.append(f"{'model':<6} {'persona':<18} {'scenario':<24} {'words/reply':>11} {'spoken s':>9} {'leak':>5}")
    lines.append("-" * 80)
    for (model_alias, persona, scenario), records in sorted(cells.items()):
        replies = [r["reply"] for r in records]
        words = sum(len(reply.split()) for reply in replies) / max(1, len(replies))
        spoken = words / SPOKEN_WORDS_PER_SECOND
        leak = any(tool_syntax_leak(reply) for reply in replies)
        lines.append(
            f"{model_alias:<6} {persona:<18} {scenario:<24} {words:>11.1f} {spoken:>9.1f} {'YES' if leak else '-':>5}"
        )
    lines.append("\n\n## Transcripts\n")
    for (model_alias, persona, scenario), records in sorted(cells.items()):
        lines.append(f"\n### {model_alias} / {persona} / {scenario}\n")
        for record in records:
            speaker = "SYSTEM->" if record["role"] == "intervention" else "STUDENT:"
            lines.append(f"- **{speaker}** {record['text']}")
            lines.append(f"  - **REACHY** ({record['output_tokens']} tok, {record['seconds']}s): {record['reply']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true", help="rebuild report from cached cells only")
    parser.add_argument("--models", help="comma-separated model aliases to run (default: all)")
    parser.add_argument("--personas", help="comma-separated persona names to run (default: all)")
    args = parser.parse_args()

    models = MODELS
    if args.models:
        wanted = args.models.split(",")
        unknown = [alias for alias in wanted if alias not in MODELS]
        if unknown:
            sys.exit(f"unknown model alias(es): {', '.join(unknown)} (have: {', '.join(MODELS)})")
        models = {alias: MODELS[alias] for alias in wanted}

    personas = {path.stem: path.read_text() for path in sorted((HERE / "personas").glob("*.txt"))}
    if args.personas:
        wanted = args.personas.split(",")
        unknown = [name for name in wanted if name not in personas]
        if unknown:
            sys.exit(f"unknown persona(s): {', '.join(unknown)} (have: {', '.join(personas)})")
        personas = {name: personas[name] for name in wanted}
    scenarios = {
        name: turns
        for name, turns in json.loads((HERE / "scenarios" / "scenarios.json").read_text()).items()
        if not name.startswith("_")
    }
    cache_dir = HERE / "results_cache"
    cache_dir.mkdir(exist_ok=True)

    cells = {}
    for model_alias, (model, _tps) in models.items():
        for persona, instructions in personas.items():
            for scenario, turns in scenarios.items():
                cell_path = cache_dir / f"{model_alias}__{persona}__{scenario}.json"
                if cell_path.exists():
                    cells[(model_alias, persona, scenario)] = json.loads(cell_path.read_text())
                    continue
                if args.report_only:
                    continue
                print(f"running {model_alias} / {persona} / {scenario} ...", flush=True)
                try:
                    records = run_cell(model, instructions, turns)
                except Exception as error:
                    print(f"  FAILED: {error}", file=sys.stderr)
                    continue
                cell_path.write_text(json.dumps(records, indent=1))
                cells[(model_alias, persona, scenario)] = records

    report_path = HERE / "results_report.md"
    report_path.write_text(build_report(cells))
    print(f"\n{len(cells)} cells -> {report_path}")


if __name__ == "__main__":
    main()
