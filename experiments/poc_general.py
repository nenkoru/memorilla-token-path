"""H13: is the biased core + reader a general memory, or only good for code names?

Synthetic "people in the user's life" facts ("Priya Nair lives in Lisbon."), one fact per chunk, direct questions
("Which city does Priya Nair live in?"). The pile holds the facts of N people and everything passes through ONE
16-token block, as in H12 (attention logit += beta * cos(question, fact)). Key vector = value vector = the fact.

Held out for the test: people (first and last names never seen in training), a quarter of every value list, and three
whole attributes (instrument, drink, hometown).

    gen     general.json (no model)
    embed   general_emb.npz: fact and question vectors (Qwen3-Embedding-4B)
    cos     does the encoder's cosine alone pick the asked fact? (CPU)
    train   init lora_bias.pt + beta from core_bias.pt; 60% people facts, 40% code facts (H12 layout, plain values)
    eval    piles of N people, plus N=16 with the asked fact removed; lines starting GEN
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

FIRST = """Marcus Elena Tomas Aisha Kenji Sofia Dmitri Hannah Omar Ingrid Rafael Mei Callum Zara Viktor Naomi Pablo Freya
Idris Chloe Anders Leila Mateo Yuki Gareth Amara Stefan Nadia Luca Esther Farid Greta Hugo Imani Jonas Katya Liam Mira
Nikhil Olga Pedro Quinn Rosa Samir Tess Ulrich Vera Wendell Ximena Yusuf Zoe Arjun Bianca Cedric Daria Emil Fatima Goran
Helga Ivan Jada Karim Lena Milan Nora Oskar Paloma Rhys Selma Tariq Uma Vince Willa Xavier Yara Zane Alma Boris Celia Dev
Edith Felix Gia Hamid Iris Joel Kira Lars Maeve Nico Opal Pavel Ruth Soren Talia Uri Violet Wyatt Priya Lila""".split()
LAST = """Okafor Lindqvist Moreau Tanaka Castillo Novak Haddad Brennan Kowalski Mensah Ferrari Sandoval Petrov Yamada
Gallagher Dubois Rahman Schmidt Larsen Popescu Achebe Romero Fischer Horvat Silva Jensen Kaplan Demir Vasquez Olsen
Bianchi Sharma Kovacs Abdi Navarro Hoffman Suzuki Marchetti Eriksen Baptiste Volkov Chaudhry Lemaire Osei Pereira Winter
Zhou Markovic Aguilar Thorsen Banerjee Costa Hale Ibarra Jovanovic Klein Laurent Moyo Nilsson Nair""".split()
N_TEST_FIRST, N_TEST_LAST = 20, 12  # taken from the END of the lists (so Priya, Lila and Nair are test-only)

CITIES = """Lisbon; Oslo; Krakow; Nairobi; Osaka; Montevideo; Tbilisi; Hanoi; Porto; Ghent; Tampere; Valencia; Split;
Leipzig; Cork; Bergen; Graz; Lyon; Bologna; Malmo; Riga; Tartu; Kyoto; Busan; Chiang Mai; Medellin; Cusco; Halifax;
Tucson; Boise; Durban; Accra; Tunis; Amman; Pune; Kochi; Perth; Hobart; Dunedin; Wellington; Quebec City; Savannah;
Madison; Aarhus; Utrecht; Basel; Salzburg; Brno; Plovdiv; Izmir; Baku; Almaty; Colombo; Da Nang; Cebu; Fukuoka;
Sapporo; Recife; Mendoza; Windhoek"""
JOBS = """nurse; pilot; welder; architect; pharmacist; translator; electrician; pastry chef; tax accountant; librarian;
paramedic; florist; locksmith; radiologist; surveyor; beekeeper; tattoo artist; bus driver; court reporter; glassblower;
midwife; carpenter; sommelier; geologist; dental hygienist; sound engineer; park ranger; tailor; actuary; zookeeper;
ferry captain; speech therapist; stonemason; optician; crane operator; museum curator; watchmaker; plumber;
veterinarian; typesetter"""
HOBBIES = """rock climbing; pottery; birdwatching; chess; fly fishing; knitting; salsa dancing; woodworking; astronomy;
sourdough baking; kayaking; calligraphy; falconry; trail running; origami; archery; model trains; fencing; gardening;
crossword puzzles; surfing; embroidery; foraging; karaoke; judo; stamp collecting; home brewing; improv comedy;
scuba diving; quilting; table tennis; bonsai; geocaching; roller skating; wood carving; curling; snowboarding;
bookbinding; paragliding; lock picking"""
FOODS = """pad thai; lasagna; ramen; falafel; paella; pierogi; bibimbap; moussaka; ceviche; goulash; tacos al pastor;
shakshuka; pho; risotto; dumplings; jollof rice; borscht; tiramisu; gnocchi; biryani; empanadas; okonomiyaki;
khachapuri; poutine; laksa; baklava; clam chowder; fish tacos; mapo tofu; schnitzel; tabbouleh; carbonara; katsu curry;
arepas; ratatouille; hummus; banh mi; fondue; gumbo; saag paneer"""
PETS = """Biscuit; Mochi; Pepper; Waffles; Juniper; Ziggy; Nacho; Olive; Sprout; Bandit; Maple; Clover; Pickle; Dumpling;
Cosmo; Tofu; Figaro; Peanut; Noodle; Hazel; Rocket; Sage; Truffle; Gizmo; Pretzel; Luna; Basil; Pumpkin; Marble; Otis;
Wasabi; Birdie; Chai; Domino; Echo; Fennel; Ginger; Hopper; Indigo; Jellybean"""
RELATIONS = """sister; brother; dentist; landlord; manager; neighbour; cousin; yoga teacher; college roommate; accountant;
mentor; aunt; uncle; barber; personal trainer; mother-in-law; godfather; piano teacher; business partner; old classmate;
doctor; lawyer; babysitter; climbing partner; stepbrother; niece; nephew; tutor; bandmate; editor"""
ALLERGIES = """peanuts; shellfish; penicillin; latex; cat hair; pollen; gluten; eggs; soy; bee stings; dust mites;
strawberries; sesame; mould; kiwi; ibuprofen; nickel; lactose; walnuts; mustard; celery; wool; aspirin; mango"""
LANGS = """Portuguese; Japanese; Swahili; Polish; Korean; Arabic; Finnish; Turkish; Hindi; Dutch; Greek; Mandarin; Welsh;
Icelandic; Vietnamese; Hebrew; Czech; Tagalog; Hungarian; Thai; Norwegian; Persian; Catalan; Georgian"""
INSTRUMENTS = """cello; banjo; trumpet; accordion; oboe; ukulele; harp; clarinet; drums; mandolin; saxophone; flute;
bagpipes; violin; trombone; bass guitar; harmonica; viola; tuba; sitar"""
DRINKS = """oat flat white; green tea; espresso; ginger beer; black coffee; chai latte; sparkling water; hot chocolate;
matcha latte; lemonade; iced americano; peppermint tea; kombucha; cortado; orange juice; earl grey; cappuccino;
tonic water; cold brew; mango lassi"""
COLOURS = "red; blue; silver; white; black; green; grey; yellow; orange; dark blue"
MODELS = """Toyota Corolla; Honda Civic; Volkswagen Golf; Ford Fiesta; Mazda 3; Subaru Outback; Skoda Octavia; Kia Sportage;
Fiat Panda; Renault Clio; Nissan Leaf; Volvo V60; Peugeot 208; Hyundai Tucson; Tesla Model 3; Mini Cooper; Jeep Wrangler;
Opel Corsa; Seat Ibiza; Suzuki Swift; Dacia Duster; Audi A4; BMW i3; Citroen C3"""
CO_A = "Northwind; Bluepeak; Harrow; Kestrel; Lumen; Orchard; Quarry; Saffron; Tidewater; Vantage; Wren; Zephyr; Copperline; Driftwood; Elmstead; Foxglove"
CO_B = "Logistics; Labs; Dental; Foods; Robotics; Insurance; Press; Energy; Studios; Freight"
MONTHS = "January February March April May June July August September October November December".split()


def lst(s):
    return [x.strip() for x in s.replace("\n", " ").split(";") if x.strip()]


def pairs(a, b):
    return [f"{x} {y}" for x in lst(a) for y in lst(b)]


# attribute -> (fact, question, values, held-out attribute?)
ATTRS = {
    "city": ("{p} lives in {v}.", "Which city does {p} live in?", lst(CITIES), False),
    "job": ("{p}'s job is {v}.", "What is {p}'s job?", lst(JOBS), False),
    "hobby": ("{p}'s main hobby is {v}.", "What is {p}'s main hobby?", lst(HOBBIES), False),
    "food": ("{p}'s favourite food is {v}.", "What is {p}'s favourite food?", lst(FOODS), False),
    "pet": ("{p} has a pet called {v}.", "What is {p}'s pet called?", lst(PETS), False),
    "birthday": ("{p}'s birthday is on {v}.", "When is {p}'s birthday?", [f"{m} {d}" for m in MONTHS for d in range(1, 29)], False),
    "car": ("{p} drives a {v}.", "What car does {p} drive?", pairs(COLOURS, MODELS), False),
    "relation": ("{p} is the user's {v}.", "Who is {p} to the user?", lst(RELATIONS), False),
    "employer": ("{p} works at {v}.", "Which company does {p} work at?", pairs(CO_A, CO_B), False),
    "allergy": ("{p} is allergic to {v}.", "What is {p} allergic to?", lst(ALLERGIES), False),
    "language": ("{p} is learning {v}.", "Which language is {p} learning?", lst(LANGS), False),
    "instrument": ("{p} plays the {v}.", "Which instrument does {p} play?", lst(INSTRUMENTS), True),
    "drink": ("{p}'s usual drink is {v}.", "What is {p}'s usual drink?", lst(DRINKS), True),
    "hometown": ("{p} grew up in {v}.", "Where did {p} grow up?", lst(CITIES), True),
}
SUFFIX = " Answer in a few words."


def out_dir():
    from poc_agentmem import OUT
    return OUT


def cmd_gen(args):
    rng = random.Random(0)
    pools = {}
    for a, (_, _, vals, _) in ATTRS.items():
        v = vals[:]
        random.Random(len(a)).shuffle(v)
        pools[a] = (v[: int(len(v) * 0.75)], v)  # (values training may use, all values)
    names = {"train": [f"{f} {l}" for f in FIRST[:-N_TEST_FIRST] for l in LAST[:-N_TEST_LAST]],
             "test": [f"{f} {l}" for f in FIRST[-N_TEST_FIRST:] for l in LAST[-N_TEST_LAST:]]}
    train_attrs = [a for a, x in ATTRS.items() if not x[3]]
    held = [a for a, x in ATTRS.items() if x[3]]
    people = []
    for split, n in (("train", args.train), ("test", args.test)):
        for name in rng.sample(names[split], n):
            attrs = rng.sample(train_attrs, 6) if split == "train" else rng.sample(train_attrs, 5) + held
            facts = []
            for a in attrs:
                v = rng.choice(pools[a][0] if split == "train" else pools[a][1])
                facts.append({"attr": a, "value": v, "fact": ATTRS[a][0].format(p=name, v=v),
                              "question": ATTRS[a][1].format(p=name) + SUFFIX,
                              "held_attr": ATTRS[a][3], "value_seen": v in pools[a][0]})
            people.append({"name": name, "split": split, "facts": facts})
    (out_dir() / "general.json").write_text(json.dumps(people, indent=1))
    nf = sum(len(p["facts"]) for p in people)
    print(f"people {len(people)}  facts {nf}", flush=True)
    for p in people[:2] + people[-2:]:
        print(p["split"], [f["fact"] for f in p["facts"]], flush=True)


def load():
    people = json.loads((out_dir() / "general.json").read_text())
    o = 0
    for p in people:
        p["rows"] = np.arange(o, o + len(p["facts"]))
        o += len(p["facts"])
    emb = out_dir() / "general_emb.npz"
    if not emb.exists():
        return people, None, None
    z = np.load(emb)
    assert len(z["facts"]) == o
    return people, z["facts"], z["q"]


def cmd_embed(args):
    import torch
    from poc_agentmem import D
    people, _, _ = load()
    facts = [f["fact"] for p in people for f in p["facts"]]
    qs = [f["question"] for p in people for f in p["facts"]]
    enc = D.Encoder()
    t0, out, texts = time.perf_counter(), [], facts + qs
    for b in range(0, len(texts), 512):
        out.append(enc(texts[b:b + 512], batch_size=32).astype(np.float16))
        print(f"{b + len(out[-1])}/{len(texts)}  {time.perf_counter() - t0:.0f}s", flush=True)
        torch.mps.empty_cache()
    v = np.concatenate(out)
    np.savez(out_dir() / "general_emb.npz", facts=v[: len(facts)], q=v[len(facts):])
    print("done", flush=True)


def cmd_cos(args):
    people, F, Q = load()
    norm = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
    F, Q = norm(F.astype(np.float32)), norm(Q.astype(np.float32))
    test = np.concatenate([p["rows"] for p in people if p["split"] == "test"])
    for label, pool in (("test facts", test), ("all facts", np.arange(len(F)))):
        sim = Q[test] @ F[pool].T
        own = np.searchsorted(pool, test)
        top1 = float((sim.argmax(1) == own).mean())
        line = f"COS among {len(pool)} {label}: asked fact ranked first {top1:.3f}"
        for beta in (50, 90):
            e = np.exp(beta * (sim - sim.max(1, keepdims=True)))
            line += f"  mass at beta {beta} {float((e[np.arange(len(test)), own] / e.sum(1)).mean()):.2f}"
        print(line, flush=True)


def build_pile(target, others, drop_row=None):
    rows = np.concatenate([target["rows"]] + [o["rows"] for o in others])
    return rows if drop_row is None else rows[rows != drop_row]


def cmd_train(args):
    import torch
    import poc_agentmem as A
    import poc_core as C
    import poc_spell as S
    from poc_agentmem import P, OUT, save
    torch.manual_seed(0)
    people, F, Q = load()
    train = [p for p in people if p["split"] == "train"]
    ss, ckeys, cq, cplain = C.load_all()
    ctrain = [s for s in ss if s["split"] == "train"]
    st = A.Stack("lora", train=True)
    C.load_lora(st, args.init)
    core = st.core.float().eval()
    lps = [t for m in st.lora for t in (m.A, m.B)]
    beta = torch.nn.Parameter(torch.load(OUT / args.core, map_location="cpu")["beta"].float().to(P.DEV))
    opt = torch.optim.AdamW([{"params": lps, "lr": args.lr, "base": args.lr}, {"params": [beta], "lr": 0.5, "base": 0.5}], weight_decay=0.0)
    rng, nrng = random.Random(0), np.random.default_rng(0)
    dt = next(st.r.model.parameters()).dtype
    sizes, w = [1, 2, 4, 8, 16, 32, 64], [0.1, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1]
    seen = {"people": 0, "code": 0}

    def loss_one():
        if rng.random() < 0.6:
            seen["people"] += 1
            p = rng.choice(train)
            j = rng.randrange(len(p["facts"]))
            others = rng.sample([o for o in train if o is not p], rng.choices(sizes, w)[0] - 1)
            rows = nrng.permutation(build_pile(p, others))
            mem = C.memory(core, rows, F, F, Q[p["rows"][j]], beta)
            question, answer = p["facts"][j]["question"], p["facts"][j]["value"]
        else:
            seen["code"] += 1
            s = rng.choice(ctrain)
            kind = rng.choice([k for k in S.KINDS if any(fk == k for fk, _ in s["facts"])])
            others = rng.sample([o for o in ctrain if o is not s], rng.choices([1, 2, 4, 8, 16, 32], [0.15, 0.15, 0.2, 0.2, 0.15, 0.15])[0] - 1)
            rows, _ = C.pile(s, others, nrng)
            mem = C.memory(core, rows, ckeys, cplain, cq[s["si"], S.KINDS.index(kind)], beta)
            question, answer = C.QK[kind].format(t=C.title(s)), ", ".join(x for k, x in s["facts"] if k == kind)
        mem = mem.to(dt)
        ids, labels = st.ids(question, mem.shape[1], answer)
        emb = P.input_embeds(st.r.model, st.r.tok, st.r.memory_id, ids, mem)
        return st.r.model(inputs_embeds=emb, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss

    def checkpoint():
        torch.save(st.state(), OUT / f"lora_{args.tag}.pt")
        torch.save({"beta": beta.detach().cpu()}, OUT / f"core_{args.tag}.pt")

    st.r.model.train()
    t0, u, run = time.perf_counter(), 0, []
    budget, next_save = args.minutes * 60, 1500
    while time.perf_counter() - t0 < budget:
        frac = min(1.0, (time.perf_counter() - t0) / budget)
        for g in opt.param_groups:
            g["lr"] = g["base"] * min(1.0, (u + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for _ in range(4):
            loss = loss_one() / 4
            loss.backward()
            run.append(loss.item() * 4)
        torch.nn.utils.clip_grad_norm_(lps, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        u += 1
        torch.mps.empty_cache()
        if u % 25 == 0:
            b = " ".join(f"{x:.0f}" for x in beta.detach().mean(1).tolist())
            print(f"update {u}  examples {u * 4}  loss {np.mean(run[-100:]):.3f}  {(time.perf_counter() - t0) / 60:.0f} min  beta {b}", flush=True)
        if time.perf_counter() - t0 > next_save:
            next_save += 1500
            checkpoint()
    checkpoint()
    save(f"general_train_{args.tag}.json", {"updates": u, "examples": u * 4, "loss": float(np.mean(run[-100:])),
                                            "minutes": args.minutes, "seen": seen, "beta": beta.detach().cpu().tolist()})
    print("saved", flush=True)


def cmd_eval(args):
    import torch
    import poc_agentmem as A
    import poc_core as C
    from poc_agentmem import P, OUT, save
    people, F, Q = load()
    test = [p for p in people if p["split"] == "test"]
    rest = [p for p in people if p["split"] != "test"]
    st = A.Stack("lora", train=True)
    C.load_lora(st, args.lora, eval_mode=True)
    core = st.core.float().eval()
    beta = torch.load(OUT / args.core, map_location="cpu")["beta"].float().to(P.DEV)
    if args.beta is not None:
        beta = torch.full_like(beta, args.beta)
    st.r.model.eval()
    dt = next(st.r.model.parameters()).dtype
    rng = random.Random(5)
    items = [(p, j) for p in test for j in range(len(p["facts"]))]
    rng.shuffle(items)
    conds = [(int(x), True) for x in args.sizes.split(",")] + [(16, False)]
    rows_out, t0, tab = [], time.perf_counter(), {}
    with torch.no_grad():
        for i, (p, j) in enumerate(items[: args.n]):
            f, row = p["facts"][j], p["rows"][j]
            r2 = random.Random(i)
            for n, inc in conds:
                others = r2.sample([o for o in test if o is not p], min(len(test) - 1, n - 1))
                if n - 1 > len(others):
                    others += r2.sample(rest, min(len(rest), n - 1 - len(others)))
                rows = np.random.default_rng(i).permutation(build_pile(p, others, None if inc else row))
                mem, attns = C.memory(core, rows, F, F, Q[row], beta, want_attn=True)
                ans = A.generate(st, f["question"], mem.to(dt), 24).strip()
                rows_out.append({"person": p["name"], "attr": f["attr"], "held_attr": f["held_attr"], "value_seen": f["value_seen"],
                                 "condition": f"N={n}" if inc else f"N={n} asked fact removed", "n_facts": int(len(rows)),
                                 "truth": f["value"], "answer": ans, "hit": float(f["value"].lower() in ans.lower()),
                                 "attn_asked": float(attns[1][:, rows == row].sum(-1).mean())})
            torch.mps.empty_cache()
            if (i + 1) % 10 == 0 or i + 1 == min(args.n, len(items)):
                tab = {}
                for r in rows_out:
                    t = tab.setdefault(r["condition"], {"all": [], "trained attr": [], "held-out attr": [], "unseen value": [], "attn": [], "facts": []})
                    t["all"].append(r["hit"])
                    t["held-out attr" if r["held_attr"] else "trained attr"].append(r["hit"])
                    if not r["value_seen"]:
                        t["unseen value"].append(r["hit"])
                    t["attn"].append(r["attn_asked"])
                    t["facts"].append(r["n_facts"])
                tab = {c: {k: (float(np.mean(v)) if v else None) for k, v in t.items()} for c, t in tab.items()}
                save(args.out, {"lora": args.lora, "core": args.core, "beta": beta.cpu().tolist(), "rows": rows_out, "table": tab})
                print(f"{i + 1}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    for c, t in tab.items():
        print(f"GEN {c:26s} facts {t['facts']:.0f}  all {t['all']:.2f}  trained attr {t['trained attr']:.2f}  "
              f"held-out attr {t['held-out attr']:.2f}  unseen value {t['unseen value']:.2f}  attention on asked fact {t['attn']:.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--train", type=int, default=500)
    g.add_argument("--test", type=int, default=200)
    sub.add_parser("embed")
    sub.add_parser("cos")
    t = sub.add_parser("train")
    t.add_argument("--minutes", type=float, default=75)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--init", default="lora_bias.pt")
    t.add_argument("--core", default="core_bias.pt")
    t.add_argument("--tag", default="gen")
    e = sub.add_parser("eval")
    e.add_argument("--n", type=int, default=150)
    e.add_argument("--lora", default="lora_gen.pt")
    e.add_argument("--core", default="core_gen.pt")
    e.add_argument("--beta", type=float, default=None)
    e.add_argument("--sizes", default="1,16,200,700")
    e.add_argument("--out", default="general_eval.json")
    args = ap.parse_args()
    {"gen": cmd_gen, "embed": cmd_embed, "cos": cmd_cos, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
