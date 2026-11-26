"""A second score for the buried-clue answers: production's 27B reads the request, the fact and the reply, and says whether
the reply takes the fact into account. The keyword rubric (acts_strict) is the other score. No GPU of mine is used.

    judge27.py --url http://127.0.0.1:8001 --model <name> name=file[+file2] ...     files under work/runs/poc
Judgments are cached in judge27.json by (question, reply), so a rerun only asks for new replies.
    judge27.py --url ... --model ... --rows selfgen_0.json+selfgen_1.json --keep selfkeep.json
Rows with request, relevant and answer (poc_selfgen.py): every reply judged on its relevant fact; per fact set the reply with
the highest Yes probability is kept when above 0.5 (rows for poc_broad.py train --mix27).

How good a score is, without human labels: a reply written without the fact cannot take it into account, so every Yes on
"clue removed" is a false alarm. Printed per reader: the score with the clue alone minus the score with the clue removed.
"""
import argparse
import hashlib
import json
import math
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_broad as B  # noqa: E402
import poc_profile as PF  # noqa: E402
import poc_whole as W  # noqa: E402

SYSTEM = "You grade replies written by an assistant. Answer with one word: Yes or No."
ASK = ("A user wrote to an assistant:\n{q}\n\nThe assistant had this fact about the user in its memory:\n{f}\n\n"
       "The assistant's reply (it may be cut off at the end):\n{a}\n\n"
       "Does the reply take the fact into account? Say Yes only if the reply is shaped by the fact, so that it would have to be "
       "different for a user without it, and nothing in the reply goes against the fact. A reply that would suit any user is No.")


def key(pair, answer):
    return pair + "|" + hashlib.sha1(answer.encode()).hexdigest()[:16]


def p_yes(choice):
    """Probability of Yes among Yes and No at the first generated token."""
    top = (choice.get("logprobs") or {}).get("content") or []
    yes = no = 0.0
    for t in (top[0]["top_logprobs"] if top else []):
        w = t["token"].strip().lower()
        if w.startswith("yes"):
            yes += math.exp(t["logprob"])
        elif w.startswith("no"):
            no += math.exp(t["logprob"])
    if yes + no > 0:
        return yes / (yes + no)
    return float((choice["message"]["content"] or "").strip().lower().startswith("yes"))


def ask(url, model, q, f, a):
    body = {"model": model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 8,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASK.format(q=q.strip(), f=f.strip(), a=a.strip())}]}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return p_yes(json.loads(urllib.request.urlopen(req, timeout=1800).read())["choices"][0])


def judge_rows(args, K):
    url = args.url.rstrip("/")
    rows = [r for f in args.rows.split("+") for r in json.loads((K.OUT / f).read_text())["rows"]]
    cache_p = K.OUT / "judge27.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    kk = [key("self:" + r["request"] + "|" + r["relevant"], r["answer"]) for r in rows]
    todo = [i for i, k in enumerate(kk) if k not in cache]
    print(f"JUDGE rows: {len(rows)} replies, {len(todo)} to judge", flush=True)
    t0 = time.perf_counter()
    for b in range(0, len(todo), args.batch):
        while W._waiting(url) > 0:  # someone else is queued on the shared server: let them go first
            time.sleep(2)
        part = todo[b:b + args.batch]
        with ThreadPoolExecutor(len(part)) as ex:
            for i, p in zip(part, ex.map(lambda i: ask(url, args.model, rows[i]["request"], rows[i]["relevant"], rows[i]["answer"]), part)):
                cache[kk[i]] = round(p, 4)
        if (b // args.batch) % 50 == 0 or b + args.batch >= len(todo):
            cache_p.write_text(json.dumps(cache))
            print(f"JUDGE {min(b + args.batch, len(todo))}/{len(todo)} at +{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    sets = {}
    for r, k in zip(rows, kk):
        r["judge"] = cache[k]
        sets.setdefault((r["request"], tuple(r["facts"])), []).append(r)
    keep = [max(g, key=lambda r: r["judge"]) for g in sets.values()]
    keep = [r for r in keep if r["judge"] > 0.5]
    first = [g[0]["judge"] > 0.5 for g in sets.values()]
    anyy = [max(r["judge"] for r in g) > 0.5 for g in sets.values()]
    print(f"JUDGE rows: {len(sets)} fact sets; Yes for one sample {np.mean([r['judge'] > 0.5 for r in rows]):.2f}, "
          f"first sample {np.mean(first):.2f}, best of the set {np.mean(anyy):.2f}; kept {len(keep)}", flush=True)
    (K.OUT / args.keep).write_text(json.dumps({"rows": keep}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("readers", nargs="*")
    ap.add_argument("--rows", default="")
    ap.add_argument("--keep", default="selfkeep.json")
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    K = PF.poc()
    if args.rows:
        return judge_rows(args, K)
    su, qvec, _, _ = B.load_all()
    ev, _ = B.test_set(su, qvec, True)
    ent = {e["pair"]: e for e in ev}
    readers = []
    for spec in args.readers:
        name, files = spec.split("=")
        rows = {}
        for f in files.split("+"):
            for r in json.loads((K.OUT / f).read_text())["rows"]:
                if r["group"] == "held-out types":
                    rows[(r["condition"].replace("tokLr27w_full_change ", ""), r["pair"])] = r
        readers.append((name, rows))
    cache_p = K.OUT / "judge27.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    todo = list({key(r["pair"], r["answer"]): r for _, rows in readers for r in rows.values() if key(r["pair"], r["answer"]) not in cache}.values())
    todo = todo[: args.limit or None]
    print(f"JUDGE {sum(len(rows) for _, rows in readers)} replies, {len(todo)} to judge, {len(cache)} cached", flush=True)
    url = args.url.rstrip("/")

    def one(r):
        e = ent[r["pair"]]
        body = {"model": args.model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 8,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": ASK.format(q=e["question"].strip(), f=e["clue"].strip(), a=r["answer"].strip())}]}
        req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        return p_yes(json.loads(urllib.request.urlopen(req, timeout=1800).read())["choices"][0])

    t0 = time.perf_counter()
    for b in range(0, len(todo), args.batch):
        while W._waiting(url) > 0:  # someone else is queued on the shared server: let them go first
            time.sleep(2)
        part = todo[b:b + args.batch]
        with ThreadPoolExecutor(len(part)) as ex:
            for r, p in zip(part, ex.map(one, part)):
                cache[key(r["pair"], r["answer"])] = round(p, 4)
        if (b // args.batch) % 50 == 0 or b + args.batch >= len(todo):
            cache_p.write_text(json.dumps(cache))
            print(f"JUDGE {min(b + args.batch, len(todo))}/{len(todo)} at +{(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    def both(rows, c, pairs):
        got = [(rows[(c, p)], cache.get(key(p, rows[(c, p)]["answer"]))) for p in pairs]
        got = [(r, j) for r, j in got if j is not None]
        return np.array([r["acts_strict"] for r, _ in got], dtype=float), np.array([j for _, j in got]) > 0.5, np.array([j for _, j in got])

    conds = list(dict.fromkeys(c for _, rows in readers for c, _ in rows))
    base_name, base = readers[0]
    print(f"JUDGE per condition, the questions every reader answered: keyword score, judge score (paired difference to {base_name} "
          f"+- one standard error), share of replies where the two scores agree", flush=True)
    for c in conds:
        have = [(n, rows) for n, rows in readers if any(cc == c for cc, _ in rows)]
        pairs = sorted(set.intersection(*[{p for cc, p in rows if cc == c} for _, rows in have]))
        pairs = [p for p in pairs if all(key(p, rows[(c, p)]["answer"]) in cache for _, rows in have)]
        if not pairs:
            continue
        line = []
        jb = both(base, c, pairs)[1].astype(float) if (c, pairs[0]) in base else None
        for name, rows in have:
            kw, j, _ = both(rows, c, pairs)
            d = j.astype(float) - jb if jb is not None and rows is not base else None
            diff = "" if d is None else f" ({d.mean():+.2f} +- {d.std(ddof=1) / len(d) ** 0.5:.2f})"
            line.append(f"{name} {kw.mean():.2f} / {j.mean():.2f}{diff} agree {np.mean(kw.astype(bool) == j):.2f}")
        print(f"JUDGE {c} (n={len(pairs)}): " + " | ".join(line), flush=True)
    print("JUDGE how well each score separates a reply that had the clue from one that did not (clue alone minus clue removed)", flush=True)
    for name, rows in readers:
        with_c = [c for c in conds if c == "clue alone" and any(cc == c for cc, _ in rows)]
        without = [c for c in conds if "removed" in c and any(cc == c for cc, _ in rows)]
        if not with_c or not without:
            continue
        a = both(rows, with_c[0], sorted(p for cc, p in rows if cc == with_c[0]))
        z = both(rows, without[0], sorted(p for cc, p in rows if cc == without[0]))
        if len(a[0]) and len(z[0]):
            print(f"JUDGE {name}: keyword {a[0].mean():.2f} - {z[0].mean():.2f} = {a[0].mean() - z[0].mean():.2f} | "
                  f"judge {a[1].mean():.2f} - {z[1].mean():.2f} = {a[1].mean() - z[1].mean():.2f} "
                  f"(judge unsure, 0.2 to 0.8: {np.mean((np.concatenate([a[2], z[2]]) > 0.2) & (np.concatenate([a[2], z[2]]) < 0.8)):.2f})", flush=True)


if __name__ == "__main__":
    main()
