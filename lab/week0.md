# Week 0: Environment Setup (asynchronous, before Session 1B)

**Goal:** Before the Session 1B lab on Wednesday, 21 Oct, you can call a large language model from your own code, with clean key handling and version control in place. If every checkbox at the end of this chapter is green, you are ready. Expect 45 to 60 minutes.

**Why this matters:** This course is about building agents, not admiring them. Everything we build from Week 1 onward assumes this environment works. Do not skip the verification steps: every single trap described below occurred in real instructor and student runs, and each one produces errors that look mysterious later but are trivial to prevent now.

**Before you start, you need three things on your laptop:**

1. **Python 3.14**, the version the course stack is validated on. Check with `python --version`. If Python is missing, install it from `https://www.python.org/downloads/` and tick "Add python.exe to PATH" in the installer.
2. **Git.** Check with `git --version`; if it is missing, see Module 0.3.
3. **An editor**, see Module 0.2. You can install it while the packages download.

> **macOS or Linux?** This chapter is validated on Windows 11 with PowerShell. Where commands differ, a short box like this one gives the equivalent. Those boxes have not been run by the instructor; if a step behaves differently on your machine, tell the TAs, and we fix the chapter for everyone.

---

## Module 0.1: Your First LLM Call (OpenRouter)

### Concept in one paragraph

Your agents need a brain. We use OpenRouter, a gateway that exposes dozens of models (Claude, DeepSeek, GPT, Llama, and more) behind one OpenAI-compatible API. One account, one key, any model. It also gives the course a single cost dashboard across all keys, which matters once teams build their capstones. Note the OpenAI compatibility itself: the OpenAI API format has become a de facto industry standard, which is why the official `openai` Python package can talk to a completely different company's gateway.

### Setup steps

Model access is funded and provisioned by the course: you do **not** create your own account or pay for anything. Your personal key arrives by e-mail on the evening of Session 1A (Monday, 19 Oct); team keys follow at the capstone launch on 28 Oct.

1. Collect your key from the course onboarding e-mail. It carries a credit limit, which is your cost ceiling for the labs; the cost check in this module shows what you have spent.
2. Copy the key using the copy button only. Do not screenshot it, do not paste it into chat tools, do not forward it.
3. Store it in `.env` as shown below, never in code and never in a commit.
4. If a key stops working or you run into its limit, write to the TAs. Running out of credit is a planning signal, not an emergency: it usually means a loop is retrying something it should not.

> **Rule 1 of key hygiene: a key that has ever appeared on a screen you shared is a compromised key.** If that happens (and it happens fast: one screenshot for a support question is enough), tell the TAs at once. Your key is disabled and you get a fresh one. Keys are cheap; leaked keys are not.

### Clone the course repository and set up Python (PowerShell)

Every script in this lab script ships in the course repository as a validated reference implementation, together with the pinned dependency versions and the frozen data snapshots. You will need Git for this step; if `git --version` fails, install it first (see Module 0.3) and come back.

```powershell
git clone https://github.com/devohp/agentic-ai-04-801-f26.git
cd agentic-ai-04-801-f26
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you have several Python versions installed and `python --version` does not show 3.14, create the venv with `py -3.14 -m venv .venv` instead.

The install pulls the full course stack, including the retrieval library used in Week 3. Doing this now, on a good connection, is the whole point of Week 0.

**How to work with the repo versions:** the scripts are printed in full in this lab script so you can read them like a book, with the surrounding autopsies and design rationale. The recommended practice each week is: read the printed version here first, then run the repo version. Retyping a script by hand is optional, occasionally enlightening, and never graded.

Your prompt must now start with `(.venv)`. If activation fails with an execution policy error, run once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

then open a new terminal and try again.

> **macOS or Linux:** `python3 -m venv .venv`, then `source .venv/bin/activate`. There is no execution policy step.

### Storing the key in `.env`

Your key lives in a file named `.env`, which stays on your machine and never enters version control (Module 0.3 verifies this). The repo ships a template; copy it and put your key in:

```powershell
copy .env.example .env
Set-Content -Path .env -Value 'OPENROUTER_API_KEY=sk-or-v1-YOUR-KEY' -Encoding ascii
Get-Content .env
```

(The `Set-Content` line overwrites the template content with your real key line; the iSDAsoil lines return in Week 3.)

The check must show exactly one line, no quotes, no spaces around the equals sign. Run `Get-Content .env` only now, alone at your desk, and never on a shared screen.

> **Trap (Notepad):** Creating the file in an editor easily produces `.env.txt` because Windows Explorer hides file extensions. Use the `Set-Content` command above, or save from Notepad with type "All files" and the name in quotes: `".env"`.
>
> **Trap (encoding):** In Windows PowerShell 5.1, `Set-Content -Encoding UTF8` writes an invisible byte order mark (BOM) at the start of the file. The BOM sticks to the variable name, and `python-dotenv` silently fails to find `OPENROUTER_API_KEY` even though the file "looks correct". This is why the command above says `-Encoding ascii`.

> **macOS or Linux:** `cp .env.example .env`, then open `.env` in your editor, replace the placeholder after the equals sign with your key, and save. Neither trap above applies.

### The smoke test

The repo already contains `hello_llm.py`. Read it before you run it; every line below earns its place:

```python
# hello_llm.py, Week 0 smoke test
# Goal: one successful LLM call via OpenRouter, with clean key handling.
# If this runs, your environment is ready for Session 1.

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads OPENROUTER_API_KEY from .env; never hardcode keys

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

response = client.chat.completions.create(
    model="deepseek/deepseek-chat-v3-0324",   # cheap and capable, our default for labs
    max_tokens=1024,                            # some providers reject calls without a budget
    messages=[
        {"role": "user",
         "content": "In one sentence: what makes an AI system 'agentic'?"}
    ],
)

print(response.choices[0].message.content)
print("---")
print(f"Tokens used: {response.usage.total_tokens} "
      f"(prompt {response.usage.prompt_tokens}, "
      f"completion {response.usage.completion_tokens})")
```

Run it:

```powershell
python hello_llm.py
```

**Expected result:** one sentence defining agentic AI, then a token line, for example:

```
Tokens used: 45 (prompt 17, completion 28)
```

### The cost check

Your key is course-issued, so you have no dashboard of your own. This one line asks OpenRouter what your key has spent so far. It never prints the key itself:

```powershell
python -c "import os, requests; from dotenv import load_dotenv; load_dotenv(); d = requests.get('https://openrouter.ai/api/v1/key', headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']}, timeout=15).json()['data']; print({k: d.get(k) for k in ('label', 'usage', 'limit', 'limit_remaining')})"
```

Real output (instructor demo key, 18 Oct 2026, after one run of `hello_llm.py` and one of the Week 1 agent; key fragment redacted):

```
{'label': 'sk-or-v1-xxx...xxx', 'usage': 0.00021925, 'limit': 1, 'limit_remaining': 0.99978075}
```

`usage` and `limit` are in US dollars, and `label` is your key in masked form. OpenRouter books with a delay of a few seconds: a check right after a run may still show the old value, so wait ten seconds and ask again.

### Three observations worth 30 seconds each

**1. Cost.** Run the cost check before and after one `python hello_llm.py`. The difference is on the order of USD 0.00004: four thousandths of a cent. Do this calculation once by hand now; when your team builds its capstone, it monitors its own budget, and you will want an intuition for what is normal.

**2. Routing.** OpenRouter is a router: the same model name may be served by different upstream providers, depending on availability and price. Your key cannot show you this; the instructor shows the Provider column of the course dashboard in Session 1B. Remember the observation; it returns in Week 2 and in Week 5, when we discuss reproducibility of agent behavior.

**3. Nondeterminism.** Run the script three times. You will get differently worded answers with different token counts, from the identical prompt at default settings. This is normal LLM behavior, not a bug. Park the observation; in Week 5 it becomes the central question of agent evaluation: how do you test a system that never says the same thing twice?

**4. Two settings you will see in every script.** The model is pinned to a dated ID, `deepseek/deepseek-chat-v3-0324`, so that every laptop in the room sees the same model; the alias `deepseek/deepseek-chat` is a moving target. And every call sets `max_tokens=1024`: measured on 27 Sep 2026, one of the providers behind the gateway rejects any call without a budget, and a budget is also a cost cap for the day a loop misbehaves. Both settings are in `hello_llm.py` above; you will meet the reason for the second one again in Week 2.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `401 Unauthorized` | Key mistyped, or `.env` not in the working directory | Re-copy the key; run the script from the folder that contains `.env` |
| `KeyError: OPENROUTER_API_KEY` | `.env` misnamed (`.env.txt`) or BOM problem | See the two traps above |
| `Activate.ps1` blocked | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, new terminal |
| Timeouts, TLS errors | Campus proxy or unstable connection | Try a different network; ask the TAs; Session 1B has an Environment Clinic |

---

## Module 0.2: Your IDE (Cursor), and Whose AI Is Whose

### Concept in one paragraph

We recommend Cursor, a VS Code fork with a built-in AI assistant; its free plan is enough for this course. Equally fine: VS Code, with or without GitHub Copilot (GitHub Education lists the current student terms). The editor choice is yours; the labs are editor-agnostic.

### The one distinction this course insists on

> **Cursor's AI helps you write the agent. OpenRouter is what your agent calls. Two different brains, two different bills.**

Cursor's assistant runs on Cursor's model credits and is a tool for building. Your agent runs on your OpenRouter key and is the thing being built. Confusing the two means misunderstanding the entire course. The good news: the distinction is empirically checkable, and you will check it now.

### Setup steps

1. Install Cursor from `https://cursor.com` (or set up VS Code).
2. **File > Open Folder**, select the `agentic-ai-04-801-f26` folder you cloned.
3. `Ctrl+Shift+P`, type "Python: Select Interpreter", choose the entry containing `.venv`. Do this even if things seem to work; see the trap below.
4. Open the built-in terminal (`` Ctrl+` ``) and verify:

```powershell
python -c "import sys; print(sys.executable)"
```

The path must end in `...\agentic-ai-04-801-f26\.venv\Scripts\python.exe` (on macOS or Linux: `.../agentic-ai-04-801-f26/.venv/bin/python`).

> **Trap (the silent wrong interpreter):** If the path points to a global Python installation instead, your script may still run, because the needed packages happen to exist globally from some earlier project. That is the dangerous version of the bug: it stays invisible until Week 3, when a package installed only in your venv suddenly produces `ModuleNotFoundError` "out of nowhere". This exact case occurred during the instructor's own validation run of this script. Check the path now, not later.

### Exercise: the orthogonality experiment

Two AI invocations that look similar and are fundamentally different. Run the cost check from Module 0.1 once before you start and note `usage`.

**Step A: use the IDE's AI.** Open `hello_llm.py`, select the top of the file, press `Ctrl+K`, and ask: *"Add a module-level docstring at the very top of the file."* Review the suggestion, then accept or reject it. Question: whose API key just paid for that? Answer: Cursor's credits. Your OpenRouter key was not touched.

**Step B: run your own agent code.** In the terminal: `python hello_llm.py`. Question: and now? Answer: your OpenRouter key.

**The proof:** Wait ten seconds, then run the cost check after Step A and again after Step B. After Step A, `usage` is unchanged; after Step B, it went up. Orthogonality demonstrated, not asserted. Afterwards, undo the docstring with `git restore hello_llm.py`, so your copy matches the repo again.

### A cautionary note on Accept, from a real run

In the instructor's validation run, the assistant was asked (less precisely) to "add a docstring explaining what this script does". It inserted a docstring block **after** the final `print` statements: syntactically legal Python (a bare string expression), runs without error, and is still wrong, because a module docstring must be the first statement in the file; `hello_llm.__doc__` would be `None`. The suggestion looked right and was wrong.

Two lessons, both of which stay with us all semester:

1. **Accept is a decision you own.** AI assistance is allowed and encouraged in this course (it is an AI course), but you are accountable for everything you submit, and you will defend it in the final live defence.
2. **Prompt precision matters.** "Add a docstring" produced a misplaced block; "add a module-level docstring at the very top of the file" produced a correct one. You will spend a good part of Week 2 on exactly this skill.

---

## Module 0.3: Version Control and the Unleakable Key (Git)

### Concept in one paragraph

Your capstone team will live in a shared GitHub repository, and your `DECISIONS.md` (the design-rationale log; you write your first entry in Week 2) lives there too. Week 0 covers the part that cannot wait: making it structurally impossible for your `.env` to enter version control. The order of commands below is deliberate: the `.gitignore` exists **before** the first `git add`, so the key is protected before Git ever looks at your files.

### Setup steps

Check Git, install from `https://git-scm.com/download/win` if missing (defaults are fine, then open a new terminal):

```powershell
git --version
```

One-time identity and line-ending configuration (Git will otherwise guess your identity and warn about line endings on every commit):

```powershell
git config --global user.name "Your Name"
git config --global user.email you@andrew.cmu.edu
git config --global core.autocrlf true
```

(The `autocrlf` setting is the standard Windows answer to the "LF will be replaced by CRLF" warning, which is harmless line-ending normalization, not an error.)

> **macOS or Linux:** Git usually comes with the system or with the Xcode command line tools. Use `git config --global core.autocrlf input` instead of `true`.

### Verify the protection in your clone

The course repo you cloned already ships a `.gitignore` that protects `.env`. Do not take that on faith; verify it:

```powershell
git status
```

**The check that matters:** you created `.env` in Module 0.1, and yet `git status` must not mention it **anywhere**: not staged, not untracked, simply invisible. Invisible means protected. If `.env` shows up, stop and fix `.gitignore` before anything else.

### The sequence you will need in Week 4 (memorize the order)

When your capstone team creates its own repository, it will not come with a `.gitignore`. The sequence below is the template; the order is the lesson. The `.gitignore` exists **before** the first `git add`, so the key is protected before Git ever looks at your files:

```powershell
git init
Set-Content -Path .gitignore -Value ".env`n.venv/`n__pycache__/" -Encoding ascii
git add .
git status          # .env must be invisible here
git commit -m "Initial commit with clean key handling"
```

### The leak drill (30 seconds, do it once)

Open `.gitignore` and delete the `.env` line. Run `git status`. Watch `.env` appear as untracked, one `git add .` away from being committed and, after a push, permanently in the repository history where removing it is genuinely painful. Restore the line. Run `git status` again; `.env` vanishes.

You have now seen both states with your own eyes. That is the point of the drill.

---

## Week 0 Sign-Off Checklist

Work through this list; every box must be green before Session 1B.

- [ ] Your course-issued OpenRouter key is in `.env` and nowhere else (no account of your own needed)
- [ ] `python hello_llm.py` prints an answer and a token line
- [ ] You ran the cost check and know what your first call cost
- [ ] You ran the script three times and observed differently worded answers (nondeterminism, parked for Week 5)
- [ ] IDE installed; interpreter check shows a path ending in `.venv\Scripts\python.exe`
- [ ] Orthogonality experiment done: Step A left `usage` unchanged, Step B raised it
- [ ] Course repo cloned; `git status` run: `.env` invisible
- [ ] Leak drill performed once
- [ ] The Week 4 init sequence read once, order understood (gitignore before add)
- [ ] 5-minute prior-knowledge self-check submitted (the form from Session 1A; the link is repeated in your onboarding e-mail)
- [ ] `python check_setup.py` printed **PASS** on every line

**Run `check_setup.py` last.** It checks, in this order: your Python version, that you run it from the repository folder, that the interpreter is the course venv, that the clone is complete, that every pinned package has the right version, that `.env` is well formed, that Git ignores it, that your key works (without spending credit), and that NASA POWER, the weather API of Week 1, is reachable. Each check ends in one PASS or FAIL line, so you never have to guess whether you are ready; a FAIL line names the step to repeat. Real output (instructor run, 18 Oct 2026, path shortened):

```
PASS  Python version: 3.14.4
PASS  Working directory is the repo
PASS  Interpreter is the course venv: C:\Users\...\agentic-ai-04-801-f26\.venv\Scripts\python.exe
PASS  Repo clone complete
PASS  Pinned packages: all versions match requirements.txt
PASS  .env file: key line present
PASS  .env ignored by Git
PASS  OpenRouter key works: credit remaining USD 1.00
PASS  NASA POWER reachable
------------------------------------------------------------
ALL PASS: you are ready for Session 1.
```

Stuck on any step? Bring it to the Environment Clinic in Session 1B, or write to the TAs. Being stuck in Week 0 costs nothing; that is precisely why the setup happens in Week 0. The clinic is a planned safety net, not a penalty box: the most expensive failure mode in a course like this is a room full of people troubleshooting instead of learning, and that is exactly what Week 0 plus the clinic exist to prevent.
