"""Demo harness for the two memory channels (Mac, MPS, Qwen3-8B + reader LoRA + frozen Memorilla core).

    profile   a few lasting facts about the user, always attached, 4 soft tokens per fact. You type an open request.
    lookup    a store of facts (1,600 facts about 200 made-up people, plus any you add) behind ONE 16-token block.

    experiments/demo.sh             interactive chat
    experiments/demo.sh examples    canned examples -> work/runs/demo/transcript.md

36 GB of RAM cannot hold the 8B decoder and the 4B embedder together, so text is embedded in a separate pass:
when you type something new, the chat saves its state, the wrapper embeds it and reloads the model (about a minute).
Anything already seen answers straight away.
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

ASK_SUFFIX = " Answer in a few words."
DO_SUFFIX = "\n\nAnswer in one or two sentences."
PROFILE = ["The user doesn't have a car.", "The user has type 1 diabetes.", "The user is vegan.", "The user has a two-year-old son.",
           "The user works night shifts as a nurse.", "The user is on a tight budget this year."]
REQUESTS = [("Plan me a weekend getaway.", 0), ("What should I pack for a long day hike?", 1),
            ("Suggest a dinner I could cook for friends on Friday.", 2), ("Plan me a relaxing Saturday.", 3),
            ("What's a good time for me to schedule a dentist appointment?", 4), ("How do I undo my last git commit?", None)]
HELP = """Commands
  <any text>          open request, answered with your profile attached (4 tokens per fact)
  /ask <question>     look a fact up in the store through one 16-token block
  /remember <fact>    add a lasting fact to your profile      /forget <n>   remove profile fact n
  /store <fact>       add a fact to the lookup store
  /profile            show the profile                        /people       sample names and questions in the store
  /examples           list canned requests                    /help  /quit"""


def out_dir():
    import pv4_port as P
    d = P.SCRATCH / "runs/demo"
    d.mkdir(parents=True, exist_ok=True)
    return d


class Cache:
    """text -> vector, filled by the embed pass."""

    def __init__(self):
        self.tp, self.vp = out_dir() / "cache_texts.json", out_dir() / "cache_vecs.npy"
        self.texts = json.loads(self.tp.read_text()) if self.tp.exists() else []
        self.vecs = np.load(self.vp) if self.vp.exists() else np.zeros((0, 2560), np.float16)
        self.idx = {t: i for i, t in enumerate(self.texts)}

    def __contains__(self, t):
        return t in self.idx

    def __getitem__(self, t):
        return self.vecs[self.idx[t]]

    def add(self, texts, vecs):
        self.texts += texts
        self.vecs = np.concatenate([self.vecs, vecs.astype(np.float16)])
        self.idx = {t: i for i, t in enumerate(self.texts)}
        np.save(self.vp, self.vecs)
        self.tp.write_text(json.dumps(self.texts))


def load_state():
    p = out_dir() / "state.json"
    s = json.loads(p.read_text()) if p.exists() else {}
    return {"profile": list(PROFILE), "store": [], "pending": [], "replay": None, **s}


def save_state(s):
    (out_dir() / "state.json").write_text(json.dumps(s, indent=1))


def canned_texts():
    return PROFILE + [r for r, _ in REQUESTS]


def cmd_embed(args):
    cache, s = Cache(), load_state()
    todo = [t for t in dict.fromkeys(canned_texts() + s["profile"] + s["store"] + s["pending"]) if t not in cache]
    if todo:
        import demos_port as D
        print(f"embedding {len(todo)} new text(s)...", flush=True)
        cache.add(todo, D.Encoder()(todo, batch_size=16))
    s["pending"] = []
    save_state(s)


def free_percent():
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=20).stdout
        return int(out.split("free percentage:")[1].split("%")[0])
    except Exception:
        return None


class Demo:
    def __init__(self, lora, core_name):
        import torch
        import poc_agentmem as A
        import poc_core as C
        import poc_general as G
        import poc_profile as R
        from poc_agentmem import P, OUT
        self.torch, self.A, self.C, self.R, self.P = torch, A, C, R, P
        self.cache = Cache()
        people, F, Q = G.load()
        self.people = [p for p in people if p["split"] == "test"]
        self.pf = [(p, j) for p in self.people for j in range(len(p["facts"]))]
        self.F = np.stack([F[p["rows"][j]] for p, j in self.pf])
        self.Q = {p["facts"][j]["question"]: Q[p["rows"][j]] for p, j in self.pf}
        self.store_text = [p["facts"][j]["fact"] for p, j in self.pf]
        self.st = A.Stack("lora", train=True)
        C.load_lora(self.st, lora, eval_mode=True)
        self.core = self.st.core.float().eval()
        self.beta = torch.load(OUT / core_name, map_location="cpu")["beta"].float().to(P.DEV)
        self.st.r.model.eval()
        self.dt = next(self.st.r.model.parameters()).dtype
        self.lora = lora

    def do(self, request, profile, max_new=90):
        """Open request with the profile attached. -> answer, memory tokens, seconds"""
        t0 = time.perf_counter()
        with self.torch.no_grad():
            if profile:
                vecs = np.stack([self.cache[f] for f in profile])
                mem = self.R.blocks(self.C, self.core, vecs, self.cache[request], 4, self.P, self.torch).to(self.dt)
            else:
                mem = self.torch.zeros(1, 0, 4096, dtype=self.dt, device=self.P.DEV)
            ans = self.A.generate(self.st, request + DO_SUFFIX, mem, max_new).strip()
        return ans, mem.shape[1], time.perf_counter() - t0

    def ask(self, question, extra=(), drop=None):
        """Lookup through one 16-token block. question must end with ASK_SUFFIX. -> answer, fact looked at, share, n, s"""
        t0 = time.perf_counter()
        qv = self.Q[question] if question in self.Q else self.cache[question]
        texts = self.store_text + list(extra)
        F = np.concatenate([self.F, np.stack([self.cache[t] for t in extra])]) if extra else self.F
        rows = np.array([i for i in range(len(texts)) if i != drop])
        with self.torch.no_grad():
            mem, attns = self.C.memory(self.core, rows, F, F, qv, self.beta, want_attn=True)
            ans = self.A.generate(self.st, question, mem.to(self.dt), 24).strip()
        a = attns[1].mean(0)
        top = int(a.argmax())
        return ans, texts[int(rows[top])], float(a[top]), len(rows), time.perf_counter() - t0


def pick_lookups(demo, n=6):
    want = [i for i, (p, j) in enumerate(demo.pf) if any(x in p["name"] for x in ("Priya", "Lila"))]
    rng = random.Random(4)
    rng.shuffle(want)
    rest = list(range(len(demo.pf)))
    rng.shuffle(rest)
    seen, out = set(), []
    for i in want + rest:
        p, j = demo.pf[i]
        if p["name"] not in seen and p["facts"][j]["attr"] not in {demo.pf[k][0]["facts"][demo.pf[k][1]]["attr"] for k in out}:
            seen.add(p["name"])
            out.append(i)
        if len(out) == n:
            break
    return out


def cmd_examples(args):
    demo = Demo(args.lora, args.core)
    L = [f"# Memory demo transcript\n\nModel: Qwen3-8B, reader `{args.lora}`, frozen Memorilla core. Generated {time.strftime('%Y-%m-%d %H:%M')}.\n",
         "## 1. Always-on profile (4 soft tokens per fact)\n", "Profile:\n"] + [f"{i + 1}. {f}" for i, f in enumerate(PROFILE)] + [""]
    for request, rel in REQUESTS:
        ans, ntok, sec = demo.do(request, PROFILE)
        L += [f"**Request:** {request}\n", f"- With the full profile ({ntok} tokens, {sec:.0f}s): {ans}"]
        if rel is not None:
            ans2, ntok2, _ = demo.do(request, [f for i, f in enumerate(PROFILE) if i != rel])
            L.append(f"- Without \"{PROFILE[rel]}\" ({ntok2} tokens): {ans2}")
        try:
            L.append(f"- With no memory at all: {demo.do(request, [])[0]}")
        except Exception as e:  # the reader was never run with zero memory tokens
            L.append(f"- With no memory at all: (not supported: {type(e).__name__})")
        L.append("")
        print(f"done: {request}", flush=True)
    L += [f"## 2. Lookup through one 16-token block ({len(demo.store_text)} facts about {len(demo.people)} made-up people)\n"]
    for n, i in enumerate(pick_lookups(demo)):
        p, j = demo.pf[i]
        f = p["facts"][j]
        ans, top, share, nf, sec = demo.ask(f["question"])
        L += [f"**Question:** {f['question'][: -len(ASK_SUFFIX)]}\n", f"- Stored fact: {f['fact']}",
              f"- Answer (16 tokens, {nf} facts in the store, {sec:.0f}s): {ans}", f"- Fact it looked at (attention {share:.2f}): {top}"]
        if n < 2:
            ans2, top2, share2, _, _ = demo.ask(f["question"], drop=i)
            L.append(f"- Same question with that fact deleted from the store: {ans2} (looked at: {top2})")
        L.append("")
        print(f"done: {f['question']}", flush=True)
    (out_dir() / "transcript.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def cmd_chat(args):
    s, cache = load_state(), Cache()
    if any(t not in cache for t in canned_texts() + s["profile"] + s["store"] + s["pending"]):
        sys.exit(7)  # the wrapper runs the embed pass and starts the chat again
    free = free_percent()
    if free is not None and free < 45 and not args.force:
        sys.exit(f"Only {free}% of memory is free; the model needs about 17 GB. Close other work (or wait for running experiments) "
                 "and try again, or pass --force.")
    print("Loading the model (about 30 s)...", flush=True)
    demo = Demo(args.lora, args.core)
    print(f"\nReady. Reader: {args.lora}. Store: {len(demo.store_text) + len(s['store'])} facts. Profile: {len(s['profile'])} facts.\n\n{HELP}\n")

    def need(text):
        if text in cache or text in demo.Q:
            return
        s["pending"].append(text)
        save_state(s)
        print("New text: it has to be embedded first, and the embedder does not fit in memory next to the model. Reloading (about a minute)...")
        sys.exit(7)

    def handle(line):
        s["replay"] = line
        if line.startswith("/remember "):
            fact = line[10:].strip()
            need(fact)
            s["profile"].append(fact)
            print(f"Profile now has {len(s['profile'])} facts ({4 * len(s['profile'])} tokens).")
        elif line.startswith("/store "):
            fact = line[7:].strip()
            need(fact)
            s["store"].append(fact)
            print(f"Store now has {len(demo.store_text) + len(s['store'])} facts.")
        elif line.startswith("/forget "):
            i = int(line[8:]) - 1
            print("Removed:", s["profile"].pop(i))
        elif line == "/profile":
            print("\n".join(f"  {i + 1}. {f}" for i, f in enumerate(s["profile"])) or "  (empty)")
        elif line == "/people":
            for p in random.Random(time.time()).sample(demo.people, 4):
                print(f"  {p['name']}: /ask {p['facts'][0]['question'][: -len(ASK_SUFFIX)]}")
        elif line == "/examples":
            print("\n".join(f"  {r}" for r, _ in REQUESTS))
        elif line in ("/help", "/?"):
            print(HELP)
        elif line.startswith("/ask "):
            q = line[5:].strip() + ASK_SUFFIX
            need(q)
            ans, top, share, nf, sec = demo.ask(q, extra=s["store"])
            print(f"{ans}\n  [16 memory tokens; {nf} facts in the store; looked at: \"{top}\" (attention {share:.2f}); {sec:.0f}s]")
        elif line.startswith("/"):
            print("Unknown command. /help lists them.")
        else:
            need(line)
            ans, ntok, sec = demo.do(line, s["profile"])
            print(f"{ans}\n  [{ntok} memory tokens for {len(s['profile'])} profile facts; {sec:.0f}s]")
        s["replay"] = None
        save_state(s)

    if s.get("replay"):
        print(f"> {s['replay']}")
        handle(s["replay"])
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in ("/quit", "/exit", "/q"):
            break
        if line:
            try:
                handle(line)
            except Exception as e:
                print("Could not do that:", e)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("embed")
    for name in ("chat", "examples"):
        c = sub.add_parser(name)
        c.add_argument("--lora", default=None)
        c.add_argument("--core", default="core_gen.pt")
        c.add_argument("--force", action="store_true")
    args = ap.parse_args()
    busy = subprocess.run(["pgrep", "-f", "poc_guard.sh"], capture_output=True, text=True).stdout.split()
    if busy and not getattr(args, "force", False) and "poc_guard.sh dm_" not in subprocess.run(
            ["pgrep", "-fl", "poc_guard.sh"], capture_output=True, text=True).stdout:
        sys.exit("An experiment is running and needs the memory; starting the demo now would get it killed. "
                 "Try again when it has finished (or pass --force to chat / examples).")
    if args.cmd != "embed" and args.lora is None:
        import pv4_port as P
        args.lora = "lora_prof.pt" if (P.SCRATCH / "runs/agentmem/lora_prof.pt").exists() else "lora_gen.pt"
    {"embed": cmd_embed, "chat": cmd_chat, "examples": cmd_examples}[args.cmd](args)


if __name__ == "__main__":
    main()
