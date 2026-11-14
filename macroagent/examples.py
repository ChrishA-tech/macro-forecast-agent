"""Run example questions on the real data and write them into README (between EXAMPLES markers)."""
from pathlib import Path

from .__main__ import DEMO
from .graph import ask, default_classifier
from .llm import default_llm
from .tools import DataProvider

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- EXAMPLES:START -->", "<!-- EXAMPLES:END -->"


def main():
    p = DataProvider.from_vendor_data()
    llm = default_llm()
    guard = "rules + learned classifier" if default_classifier() else "rules"
    lines = [f"Real data up to {p.brent['date'].iloc[-1]:%Y-%m} (Brent) and {p.macro['date'].iloc[-1]:%Y-%m} (CPI); "
             f"answers: {'Claude, grounding-checked' if llm else 'deterministic templates'}; input guard: {guard}.", ""]
    wrong = []
    for q in DEMO:
        r = ask(q, p, llm)
        attack = q.lower().startswith("ignore")
        if (r["status"] == "blocked") != attack:
            wrong.append(f"{q!r}: status {r['status']}")
        lines += [f"> **{q}**  ", f"> {r['answer']}  ",
                  f"> <sub>intent `{r['intent']}` · status `{r['status']}` · grounded `{(r['grounding'] or {}).get('passed')}`</sub>", ""]
    md = "\n".join(lines)
    if wrong:
        print(md)
        raise SystemExit("Input guard got example questions wrong; README not updated:\n  " + "\n  ".join(wrong))
    readme = ROOT / "README.md"
    s = readme.read_text(encoding="utf-8")
    readme.write_text(s[:s.index(START) + len(START)] + "\n" + md + "\n" + s[s.index(END):], encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
