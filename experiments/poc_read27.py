"""Test 51 data: the 27B writes the reader's training answers from several close facts shown as text.

Test 49: on the same picks the 27B acts on a clue that sits among 8 facts in 0.73-0.75 of the cases, the 8B in 0.48-0.49,
and the token reader (taught by the 8B) equals the 8B. This builds answers the 8B could not give: training-type requests
only (rater_train27h.json: the 27B's ratings of facts close to each request), each with 2-8 facts as text, one of which the
27B rated relevant (or none). The server is production's 27B, used as a polite client (one small batch at a time, and only
when nobody else is queued).

  poc_read27.py --url http://127.0.0.1:8001 --model <name> [--data rater_train27h.json] [--per 2] [--far 0] [--out read27.json]
  -> work/runs/poc/read27.json  rows {request, facts, scores, relevant, answer}; resumes from a partial output
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_profile as PF  # noqa: E402
import poc_whole as W  # noqa: E402


FAR = -9.0  # score written for a fact that was not rated: drawn at random, unrelated to the request


def trim(text):
    """An answer stopped by the length limit: keep it up to the last finished sentence or line."""
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s)|\n", text)]
    return text[: ends[-1]].rstrip() if ends and ends[-1] > len(text) // 2 else text


def build(rows, per, p_none, max_facts, seed, far=0.0):
    """Fact sets per request: one fact the 27B rated relevant among facts it rated not relevant (all close to the request).
    far: share of sets whose other facts are instead drawn from the whole pool (unrelated to the request, as in the
    buried-clue test, where the other facts are about the same person but not about the question); their score is FAR."""
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["q"], []).append(r)
    pool = sorted({r["f"] for r in rows})
    jobs = []
    for q in sorted(by):
        pos = sorted((r for r in by[q] if r["s"] > 1.0), key=lambda r: -r["s"])
        neg = [r for r in by[q] if r["s"] < -1.0]
        if len(neg) < max_facts:
            continue
        # the first set uses the fact rated highest; later sets prefer a relevant fact that was not written for this request
        other = [r for r in pos if not r["known"]]
        tops = ([pos[0]] + (rng.sample(other, min(per - 1, len(other))) if other else []))[:per] if pos else []
        tops = list({id(t): t for t in tops}.values())
        while pos and len(tops) < per:
            tops.append(rng.choice(pos))
        for i in range(per):
            n = rng.randint(2, max_facts)
            top = None if (not tops or rng.random() < p_none) else tops[i]
            if far and rng.random() < far:  # (no draw when far is 0: the sets stay those of read27.json)
                mine = {r["f"] for r in by[q]}
                others = [{"f": f, "s": FAR} for f in rng.sample(pool, 4 * max_facts) if f not in mine]
                pick = others[: n - (top is not None)] + ([top] if top else [])
            else:
                pick = rng.sample(neg, n - (top is not None)) + ([top] if top else [])
            rng.shuffle(pick)
            jobs.append({"request": q, "facts": [r["f"] for r in pick], "scores": [r["s"] for r in pick],
                         "relevant": top["f"] if top else None, "known": bool(top and top["known"])})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="rater_train27h.json")
    ap.add_argument("--out", default="read27.json")
    ap.add_argument("--per", type=int, default=2, help="fact sets per request")
    ap.add_argument("--p-none", type=float, default=0.15, help="share of sets with no relevant fact")
    ap.add_argument("--max-facts", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=200, help="longer answers are cut back to their last finished sentence")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--far", type=float, default=0.0, help="share of sets whose other facts are random ones from the pool, not close ones")
    args = ap.parse_args()
    from poc_clue import D, PERSONA_PROMPT
    K = PF.poc()
    jobs = build(json.loads((K.OUT / args.data).read_text()), args.per, args.p_none, args.max_facts, args.seed, args.far)[: args.limit or None]
    out_p = K.OUT / args.out
    done = json.loads(out_p.read_text())["rows"] if out_p.exists() else []
    assert all(d["request"] == j["request"] and d["facts"] == j["facts"] for d, j in zip(done, jobs)), "the partial output is from other settings"
    n_rel = sum(j["relevant"] is not None for j in jobs)
    print(f"READ27 {len(jobs)} fact sets ({n_rel} with a relevant fact, {sum(j['known'] for j in jobs)} of them written for the request;"
          f" mean {np.mean([len(j['facts']) for j in jobs]):.1f} facts), {len(done)} already answered", flush=True)
    url = args.url.rstrip("/")

    def one(j):
        body = {"model": args.model, "max_tokens": args.max_new_tokens, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": PERSONA_PROMPT},
                             {"role": "user", "content": D.user_message(list(j["facts"]), j["request"].strip())}]}
        req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        c = json.loads(urllib.request.urlopen(req, timeout=1800).read())["choices"][0]
        return (c["message"]["content"] or "").strip(), c.get("finish_reason")

    t0, cut = time.perf_counter(), 0
    for b in range(len(done), len(jobs), args.batch):
        while W._waiting(url) > 0:  # someone else is queued on the shared server: let them go first
            time.sleep(2)
        part = jobs[b:b + args.batch]
        with ThreadPoolExecutor(len(part)) as ex:
            for j, (a, fin) in zip(part, ex.map(one, part)):
                cut += fin == "length"
                done.append({**j, "answer": trim(a) if fin == "length" else a, "cut": fin == "length"})
        if b == 0 or (b // args.batch) % max(1, 96 // args.batch) == 0 or b + args.batch >= len(jobs):
            out_p.write_text(json.dumps({"rows": done}))
            print(f"READ27 {len(done)}/{len(jobs)} at +{(time.perf_counter() - t0) / 60:.1f} min; cut at the length limit {cut}", flush=True)
    out_p.write_text(json.dumps({"rows": done}))
    for d in done[:: max(1, len(done) // 3)][:3]:
        print("READ27 sample", json.dumps({k: d[k] for k in ("request", "relevant", "answer")})[:900], flush=True)
    print(f"READ27 done: {len(done)} answers, mean words {np.mean([len(d['answer'].split()) for d in done]):.0f}", flush=True)


if __name__ == "__main__":
    main()
