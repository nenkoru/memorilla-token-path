import json, sys, collections
import numpy as np
sys.path.insert(0, "experiments")
import poc_clue as K
s = K.load_synth(); dv, qmap = K.load_emb(); z = np.load(K.OUT / "keys_emb.npz")
norm = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
kv = norm(z["v"].astype(np.float32))
by = collections.defaultdict(list)
for i, d in enumerate(z["doc"]): by[str(d)].append(i)
ids = [d["doc"] for d in s["docs"]]
raw = {json.loads(l)["id"]: json.loads(l) for l in open(K.OUT / "keys_raw.jsonl")}
miss = [d for d in s["docs"] if d["kind"] == "clue" and d["doc"] not in by]
print("clue chats with no fact written:", len(miss), collections.Counter(d["type"] for d in miss))
for d in miss[:3]: print("  MISSED", d["doc"], "|", d["text"][:230].replace("\n", " "), "| raw:", repr(raw[d["doc"]]["raw"][:80]))
def hit(text, rub): return all(any(w in text.lower() for w in alt) for alt in rub["all_of"]) if "all_of" in rub else None
res = collections.defaultdict(list); ex = []
for p in s["pairs"]:
    q = norm(qmap[p["question"]].astype(np.float32)); t = p["type"]; cid = ids[p["clue"]]
    facts_txt = " ".join(f["fact"] for f in raw[cid]["facts"])
    stored = bool(raw[cid]["facts"]) and bool(hit(facts_txt, p["rubrics"]["mentions"]))
    if cid not in by:
        r, m, n = 10**6, 0.0, 0
    else:
        cand = [p["clue"]] + [i for i, d in enumerate(s["docs"]) if i != p["clue"] and ids[i] in by and d["type"] != t]
        best = []
        for c in cand:
            sims = kv[by[ids[c]]] @ q; j = int(sims.argmax()); best.append((float(sims[j]), ids[c], str(z["text"][by[ids[c]][j]])))
        sc = np.array([b[0] for b in best]); r = int((sc > sc[0]).sum()) + 1; n = len(cand)
        allk = np.concatenate([by[ids[c]] for c in cand]); own = len(by[cid])
        e = np.exp(100 * (kv[allk] @ q - (kv[allk] @ q).max())); m = float(e[:own].sum() / e.sum())
        if r > 1 and p["split"] == "test" and len(ex) < 5:
            w = max(best); ex.append((p["question"], round(best[0][0], 2), best[0][2][:60], round(w[0], 2), w[1], w[2][:60]))
    for g in (p["split"], "all", "type:" + t):
        res[g].append((r, m, n, stored))
for g, v in sorted(res.items()):
    r = np.array([x[0] for x in v])
    print(f"STRESS {g:16s} n={len(v):3d} cands {np.mean([x[2] for x in v if x[2]] or [0]):.0f}  clue stored {np.mean([x[3] for x in v]):.2f}  R@1 {np.mean(r<=1):.2f} R@3 {np.mean(r<=3):.2f} R@5 {np.mean(r<=5):.2f}  mass on clue keys {np.mean([x[1] for x in v]):.2f}")
for e in ex: print("EX", e)
qs = json.loads((K.D.DEMOS / "data/sam/questions.json").read_text())["questions"]
sam = K.sam_chats()
print("sam chats with keys:", [c["id"] for c in sam if f"sam:{c['id']}" in by], "of", len(sam))
for qd in qs:
    q = norm(qmap[qd["question"]].astype(np.float32))
    sc = {c["id"]: float((kv[by[f"sam:{c['id']}"]] @ q).max()) for c in sam if f"sam:{c['id']}" in by}
    order = sorted(sc, key=lambda k: -sc[k]); cid = qd["clue_chat"]
    print("SAMQ", qd["id"], "clue chat", cid, "rank", order.index(cid) + 1 if cid in sc else None, "of", len(order), "top3", [(k, round(sc[k], 2)) for k in order[:3]])
for cid in sorted(K.SAM_CLUES):
    print("SAMFACTS", cid, json.dumps(raw[f"sam:{cid}"]["facts"])[:420])
