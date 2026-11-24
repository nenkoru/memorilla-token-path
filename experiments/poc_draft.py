"""Check 3 (Mac quick check): draft first, then rate each 16-token block against the draft. No fact is given as text.

Test 43 rates each fact block with a hypothetical question ("could a typical reply be wrong ... because of the fact in
your memory?"). Here the plain decoder first writes a real draft reply with no memory at all, and each block is then
rated against that draft ("is this draft wrong, unsafe or unsuitable for this user because of the fact in your
memory?"). The ranking code, tables and report are poc_tok's; the new tables are tok_d<wording>.

    draft   plain decoder, no memory, one draft per request          -> runs/poc/tok_drafts.json
    rate    like poc_tok rate with the draft in the question          -> select_scores.json tables tok_dwrong, tok_dchange
Same --stride as the poc_tok runs it is compared with.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import poc_broad as B  # noqa: E402
import poc_profile as PF  # noqa: E402
import poc_tok as T  # noqa: E402

DRAFT_WORD = {
    "dwrong": 'The user asks: "{q}"\n\nA draft reply:\n"""\n{d}\n"""\n\nIs this draft wrong, unsafe or unsuitable for this user because of the fact in your memory? Answer Yes or No.',
    "dchange": 'The user asks: "{q}"\n\nA draft reply:\n"""\n{d}\n"""\n\nShould this draft be changed because of the fact in your memory? Answer Yes or No.',
}


class WithDraft:
    """Stands in for a poc_tok wording: .format(q=...) fills in the request's draft too."""

    def __init__(self, tmpl, drafts):
        self.tmpl, self.drafts = tmpl, drafts

    def format(self, q):
        return self.tmpl.format(q=q, d=self.drafts[q])


def requests(args):
    su, qvec, fvec, _ = B.load_all()
    ev, pmap = B.test_set(su, qvec, True)
    return list(dict.fromkeys(pmap[e["pair"]]["question"] for e in ev[:: args.stride]))


def cmd_draft(args):
    import torch
    from poc_agentmem import P
    from memorilla.data import apply_chat_template
    K = PF.poc()
    qs = requests(args)
    print(f"DRAFT {len(qs)} requests", flush=True)
    *_, st, core, dt = T.load_stack(args.lora)
    T.set_lora(st, False)  # the plain decoder writes the drafts
    model, tok = st.r.model, st.r.tok
    tok.padding_side = "left"
    out, t0 = {}, time.perf_counter()
    for b in range(0, len(qs), args.batch):
        chunk = qs[b:b + args.batch]
        texts = [apply_chat_template(tok, [{"role": "user", "content": q}], add_generation_prompt=True) for q in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(P.DEV)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for q, o in zip(chunk, gen):
            out[q] = re.sub(r"<think>.*?</think>", "", tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True), flags=re.S).strip()
        print(f"DRAFT {len(out)}/{len(qs)} in {time.perf_counter() - t0:.0f}s", flush=True)
    (K.OUT / "tok_drafts.json").write_text(json.dumps(out, indent=1))
    print(f"DRAFT done; example:\n{qs[0]}\n---\n{out[qs[0]][:600]}", flush=True)


def cmd_rate(args):
    K = PF.poc()
    drafts = json.loads((K.OUT / "tok_drafts.json").read_text())
    missing = [q for q in requests(args) if q not in drafts]
    assert not missing, f"{len(missing)} requests have no draft (run: poc_draft.py draft --stride {args.stride})"
    for k in args.wordings.split(","):
        T.WORD[k] = WithDraft(DRAFT_WORD[k], drafts)
    T.cmd_rate(args)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draft")
    d.add_argument("--batch", type=int, default=16)
    d.add_argument("--max-new-tokens", type=int, default=200)
    r = sub.add_parser("rate")
    r.add_argument("--wordings", default="dwrong,dchange")
    r.add_argument("--raters", default="tok")
    r.add_argument("--batch", type=int, default=8)
    for p in (d, r):
        p.add_argument("--lora", default="lora_lsort.pt")
        p.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()
    {"draft": cmd_draft, "rate": cmd_rate}[args.cmd](args)


if __name__ == "__main__":
    main()
