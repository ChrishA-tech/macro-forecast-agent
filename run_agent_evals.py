#!/usr/bin/env python3
"""
Run the agent evaluation on this Mac with free local models, commit the results, push to GitHub.

    cd ~/Downloads/macro-forecast-agent && python3 run_agent_evals.py
    python3 run_agent_evals.py --limit 20          # quick trial run (results not committed)
    python3 run_agent_evals.py --push-only

  1. checks Ollama is running and pulls the two models (~7 GB, first time only)
  2. creates a Python 3.12 environment with uv (MCP needs Python 3.10+)
  3. runs the 102-question evaluation (rules + qwen2.5:3b + qwen2.5:7b, tools over MCP; ~30-40 min)
  4. runs the tests and the regression gate, commits results/ and README.md
  5. asks for a GitHub token, confirms the account, pushes
"""
import base64
import getpass
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = HERE / ".venv312"
PY = VENV / "bin" / "python"
MODELS = ["qwen2.5:3b", "qwen2.5:7b"]
REPO = "macro-forecast-agent"
AUTHOR = ["-c", "user.name=Chrish Pep Felix A", "-c", "user.email=chrishpepfelixa@gmail.com"]
TRAILER = ("\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n"
           "Claude-Session: https://claude.ai/code/session_01DVp71MJqKBK4sk34js4Jfb")


def sh(cmd, check=True, **kw):
    r = subprocess.run([str(c) for c in cmd], cwd=HERE, text=True, **kw)
    if check and r.returncode:
        raise SystemExit(f"\nStep failed: {' '.join(map(str, cmd))}\nSend the output above to Claude.")
    return r.returncode


def api(method, path, token, body=None):
    req = urllib.request.Request("https://api.github.com" + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    for k, v in {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json", "User-Agent": "agent-evals-push"}.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    except urllib.error.URLError as e:
        return 0, {"message": f"cannot reach api.github.com ({e.reason})"}


args = sys.argv[1:]
push_only = "--push-only" in args
limit = args[args.index("--limit") + 1] if "--limit" in args else None
ok = True

if not push_only:
    # ---------------------------------------------------------------- 1. Ollama + models
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5) as r:
            have = [m["name"] for m in json.loads(r.read()).get("models", [])]
    except (urllib.error.URLError, OSError):
        raise SystemExit("Ollama is not running. Install it from https://ollama.com (Download for macOS), open the "
                         "Ollama app once, then run this script again.")
    if not shutil.which("ollama"):
        raise SystemExit("The `ollama` command is missing: open the Ollama app once (it installs the command), then retry.")
    for m in MODELS:
        if m not in have and f"{m}:latest" not in have:
            print(f"Downloading {m} (one-off) ...")
            sh(["ollama", "pull", m])

    # ---------------------------------------------------------------- 2. Python 3.12 env via uv
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    if not Path(uv).exists():
        print("Installing uv (Python installer, one-off) ...")
        sh(["sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
        uv = str(Path.home() / ".local" / "bin" / "uv")
    if not PY.exists():
        print("Creating Python 3.12 environment ...")
        sh([uv, "venv", "--python", "3.12", VENV])
    print("Installing/checking dependencies ...")
    sh([uv, "pip", "install", "-q", "--python", PY, "-r", "requirements.txt"])

    # ---------------------------------------------------------------- 3. evaluation
    cmd = [PY, "-m", "macroagent.run_evals", "--models", "rules," + ",".join(MODELS)]
    if limit:
        cmd += ["--limit", limit]
    print("\n==================== agent evaluation" + (f" (first {limit} questions)" if limit else ""))
    run_code = sh(cmd, check=False)
    print("\n==================== tests + regression gate")
    test_code = sh([PY, "-m", "pytest", "-q"], check=False)
    gate_code = sh([PY, "-m", "macroagent.eval_gate"], check=False)
    ok = run_code == 0 and test_code == 0 and gate_code == 0
    print(f"\nevaluation: {'OK' if run_code == 0 else 'FAILED'}   tests: {'OK' if test_code == 0 else 'FAILED'}   "
          f"gate: {'OK' if gate_code == 0 else 'FAILED'}")
    if limit:
        sh(["git", "checkout", "--", "README.md"], check=False)
        for f in ("agent_eval.json", "agent_eval.md", "agent_eval_records.jsonl", "traces.jsonl", "agent_eval_baseline.json"):
            (HERE / "results" / f).unlink(missing_ok=True)
        raise SystemExit("Trial run finished (nothing committed). Run without --limit for the full evaluation.")
    if run_code == 0:
        sh(["git", "add", "README.md", "results", "evals"], check=False)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=HERE).returncode:
            sh(["git", *AUTHOR, "commit", "-q", "-m", "Agent evaluation results (local models over MCP)" + TRAILER])
            print("committed results")

if not ok and input("\nSome steps failed. Push anyway? [y/N]: ").strip().lower() != "y":
    raise SystemExit("Stopped before pushing. Send the output above to Claude.")

token = getpass.getpass("\nPaste GitHub token (hidden): ").strip()
s, user = api("GET", "/user", token)
if s != 200:
    raise SystemExit(f"Token rejected ({s}): {user.get('message')}")
login = user["login"]
print(f"Authenticated as {login}")
if input(f"Push to {login}/{REPO}? [y/N]: ").strip().lower() != "y":
    raise SystemExit("Stopped. Nothing was pushed.")
api("PUT", f"/repos/{login}/{REPO}/topics", token, {"names": [
    "llm-agents", "model-context-protocol", "mcp", "llm-evaluation", "ollama", "opentelemetry", "langgraph",
    "rag", "forecasting", "guardrails"]})
hdr = base64.b64encode(f"x-access-token:{token}".encode()).decode()
r = subprocess.run(["git", "-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {hdr}", "push", "-q",
                    f"https://github.com/{login}/{REPO}.git", "main"], cwd=HERE, text=True, capture_output=True)
if r.returncode:
    raise SystemExit("git push failed:\n" + (r.stderr or r.stdout).replace(token, "<hidden>"))
print(f"\npushed https://github.com/{login}/{REPO}\nDelete the token at github.com/settings/tokens.")
