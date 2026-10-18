# check_setup.py, Week 0: are you ready for Session 1?
# Run it last, from the repository folder, with the venv active:
#     python check_setup.py
# Every line ends in PASS or FAIL. A FAIL line names the step to repeat.
# This script uses only the Python standard library, so it still runs
# when the package installation is the thing that went wrong.

import json
import subprocess
import sys
import urllib.error
import urllib.request
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parent
results = []


def report(name, ok, detail=""):
    results.append(ok)
    line = f"{'PASS' if ok else 'FAIL'}  {name}"
    print(line + (f": {detail}" if detail else ""))


def get_json(url, headers=None, timeout=20):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


# ---- 1. Python version ------------------------------------------------------
v = sys.version_info
report("Python version", v >= (3, 10),
       f"{v.major}.{v.minor}.{v.micro}" if v >= (3, 10)
       else f"{v.major}.{v.minor} found, 3.10 or newer needed. Install a newer Python, "
            "then recreate the venv (Module 0.1)")

# ---- 2. Running from the repository folder ----------------------------------
here = Path.cwd().resolve() == REPO
report("Working directory is the repo", here,
       "" if here else f"you are in {Path.cwd()}. Run: cd {REPO}")

# ---- 3. Interpreter is this repo's venv -------------------------------------
in_venv = Path(sys.prefix).resolve() == (REPO / ".venv").resolve()
report("Interpreter is the course venv", in_venv,
       sys.executable if in_venv
       else f"{sys.executable} is not {REPO / '.venv'}. Activate the venv "
            "(Module 0.1) and select it in your IDE (Module 0.2)")

# ---- 4. Repository clone is complete ----------------------------------------
needed = ["hello_llm.py", "first_agent.py", "requirements.txt", ".gitignore",
          ".env.example", "weather_snapshot.json", "soil_snapshot.json"]
missing = [f for f in needed if not (REPO / f).exists()]
is_git = (REPO / ".git").is_dir()
report("Repo clone complete", is_git and not missing,
       "" if is_git and not missing
       else ("not a git clone. " if not is_git else "")
            + (f"missing: {', '.join(missing)}. " if missing else "")
            + "Clone the repo again with git clone (Module 0.1)")

# ---- 5. Pinned packages installed -------------------------------------------
wrong = []
req = REPO / "requirements.txt"
if req.exists():
    for raw in req.read_text(encoding="utf-8").splitlines():
        spec = raw.split("#", 1)[0].strip()
        if "==" not in spec:
            continue
        name, want = (s.strip() for s in spec.split("==", 1))
        name = name.split("[", 1)[0]
        try:
            have = metadata.version(name)
            if have != want:
                wrong.append(f"{name} {have} (needs {want})")
        except metadata.PackageNotFoundError:
            wrong.append(f"{name} not installed")
report("Pinned packages", req.exists() and not wrong,
       "all versions match requirements.txt" if req.exists() and not wrong
       else ("; ".join(wrong) if wrong else "requirements.txt not found")
            + ". With the venv active, run: pip install -r requirements.txt")

# ---- 6. The .env file -------------------------------------------------------
env = REPO / ".env"
key, env_problem = None, ""
if not env.exists():
    env_problem = ("no .env file"
                   + (" (found .env.txt: the Notepad trap)" if (REPO / ".env.txt").exists() else "")
                   + ". Create it with Set-Content (Module 0.1)")
else:
    data = env.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        env_problem = "file starts with a byte order mark. Recreate it with -Encoding ascii (Module 0.1)"
    else:
        for line in data.decode("utf-8", errors="replace").splitlines():
            if line.startswith("OPENROUTER_API_KEY"):
                value = line.split("=", 1)[1] if "=" in line else ""
                if line.startswith("OPENROUTER_API_KEY=") and value.startswith("sk-or-v1-") \
                        and value == value.strip() and '"' not in value and "'" not in value:
                    key = value
                else:
                    env_problem = ("the OPENROUTER_API_KEY line is malformed: no spaces, "
                                   "no quotes, value starts with sk-or-v1- (Module 0.1)")
                break
        else:
            env_problem = "no OPENROUTER_API_KEY line. Add your key (Module 0.1)"
report(".env file", key is not None, "key line present" if key else env_problem)

# ---- 7. .env is invisible to Git --------------------------------------------
try:
    ignored = subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=REPO,
                             capture_output=True).returncode == 0
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", ".env"], cwd=REPO,
                             capture_output=True).returncode == 0
    ok = ignored and not tracked
    report(".env ignored by Git", ok,
           "" if ok else ("Git TRACKS your .env: tell a TA before you push anything" if tracked
                          else ".gitignore does not cover .env. Restore the .env line (Module 0.3)"))
except FileNotFoundError:
    report(".env ignored by Git", False, "git not found. Install Git (Module 0.3)")

# ---- 8. The key works (no credit is spent) ----------------------------------
if key is None:
    report("OpenRouter key works", False, "skipped: fix the .env line first")
else:
    try:
        status, body = get_json("https://openrouter.ai/api/v1/key",
                                {"Authorization": f"Bearer {key}"})
        d = body.get("data", {})
        rem = d.get("limit_remaining")
        report("OpenRouter key works", True,
               f"credit remaining USD {rem:.2f}" if isinstance(rem, (int, float))
               else "key accepted")
    except urllib.error.HTTPError as e:
        report("OpenRouter key works", False,
               f"HTTP {e.code}: key rejected. Re-copy it with the copy button; "
               "if it still fails, write to the TAs")
    except Exception as e:
        report("OpenRouter key works", False,
               f"no connection ({type(e).__name__}). Check your network, then try again")

# ---- 9. NASA POWER is reachable (needed from Session 1B) --------------------
try:
    status, body = get_json(
        "https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR"
        "&community=ag&latitude=-1.94&longitude=30.06&start=20260701&end=20260701&format=JSON",
        timeout=30)
    report("NASA POWER reachable", "properties" in body)
except Exception as e:
    report("NASA POWER reachable", False,
           f"{type(e).__name__}. Check your network; on campus, ask in the course channel")

# ---- Verdict ----------------------------------------------------------------
fails = results.count(False)
print("-" * 60)
print("ALL PASS: you are ready for Session 1." if fails == 0
      else f"{fails} FAIL: fix the steps named above, then run python check_setup.py again.")
sys.exit(1 if fails else 0)
