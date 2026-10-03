"""Run the 8-K study outside the notebook.

Needs eightk_lib.py in the same folder and MASSIVE_API_KEY (env var or .env).

  # 1. Cheap screen: which tags look mispriced? (implied vs realized move, no placebo)
  python run_study.py screen --tags cfo_appointment cfo_departure ceo_appointment --max-events 30

  # 2. Full study on one tag: scoreboard, placebo gap, ranking, out-of-sample
  python run_study.py study --tag cfo_appointment

  # See every valid tag
  python run_study.py tags
"""
import argparse
from pathlib import Path

import pandas as pd

import eightk_lib as L


def cmd_tags(args):
    tax = pd.DataFrame(L.api_get_all("/stocks/taxonomies/vX/disclosures", {"limit": 1000}))
    cols = ["primary_category", "secondary_category", "tertiary_category"]
    print(tax[cols].to_string(index=False))
    tax.to_csv(Path(args.out) / "taxonomy.csv", index=False)


def cmd_screen(args):
    """One number per tag: how big was the real move compared with what options implied?
    ratio ~ 1 -> priced fairly; > 1 -> market under-priced it; < 1 -> market over-priced it."""
    out = Path(args.out); out.mkdir(exist_ok=True)
    bucket = {L.BASELINE_BUCKET: L.EXPIRY_BUCKETS[L.BASELINE_BUCKET]}   # one bucket = far fewer API calls
    rows = []
    for tag in args.tags:
        print(f"\n=== {tag}")
        try:
            r = L.run_study(tag, args.start, args.end, buckets=bucket, max_events=args.max_events, label=tag)
        except Exception as e:                                          # keep scanning if one tag fails
            print(f"  skipped: {e}")
            continue
        res = r["results"]
        if res.empty:
            print("  no priced events")
            continue
        pre = res[(res.entry == "pre") & (res.otm == L.OTM_PCT) & (res.bucket == L.BASELINE_BUCKET)]
        for h in (5, 21, "exp"):
            x = pre.loc[pre.horizon == h, "ratio"].dropna()
            lo, hi = L.bootstrap_ci(x)
            rows.append({"tag": tag, "horizon": h, "n": len(x), "mean_ratio": x.mean(), "ci_lo": lo, "ci_hi": hi,
                         "median_ratio": x.median(), "share_above_1": (x > 1).mean()})
        pd.DataFrame(rows).to_csv(out / "screen.csv", index=False)      # save as we go
    df = pd.DataFrame(rows)
    if df.empty:
        print("Nothing to report.")
        return
    print("\nScreen (sorted by distance from 1.0 at h=21; wide CIs that include 1 mean 'no evidence'):")
    view = df[df.horizon == 21].assign(dist=lambda d: (d.mean_ratio - 1).abs()).sort_values("dist", ascending=False)
    print(view.drop(columns="dist").round(2).to_string(index=False))


def cmd_study(args):
    out = Path(args.out); out.mkdir(exist_ok=True)
    tag = args.tag
    print(f"In-sample {args.start}..{args.end}")
    ins = L.run_study(tag, args.start, args.end, max_events=args.max_events, label="in-sample")
    res, board = ins["results"], ins["board"]
    res.to_csv(out / f"{tag}_results.csv", index=False)         # long table: also your ML training labels
    print("\nScoreboard (mean P&L per $1 spot; * = 95% CI excludes 0):")
    print(L.fmt_board(board))

    winner = "long_call"
    if args.placebo:
        pev = L.sample_placebo(ins["events"], args.n_placebo, args.start, args.end)
        ppriced, _ = L.price_events(pev, label="placebo")
        pres = L.evaluate(ppriced)
        diff = L.difference_board(res, pres)
        diff.to_csv(out / f"{tag}_placebo_diff.csv", index=False)
        print("\nEvents minus ordinary days:")
        print(L.fmt_board(diff, value="difference"))
        ranking = L.rank_strategies(diff)
        print("\nRanking by edge over placebo:")
        print(ranking)
        winner = [k for k, v in L.STRATEGY_LABEL.items() if v == ranking.index[0]][0]
        print(f"Top strategy in-sample: {L.STRATEGY_LABEL[winner]}")

    if args.oos:
        print(f"\nOut-of-sample {L.OOS_START}..{L.OOS_END}  (run this ONCE; don't re-tune after looking)")
        oos = L.run_study(tag, L.OOS_START, L.OOS_END, label="out-of-sample")
        oos["results"].to_csv(out / f"{tag}_oos_results.csv", index=False)
        print(L.fmt_board(oos["board"]))
        cmp = board.merge(oos["board"], on=["strategy", "horizon"], suffixes=("_in", "_oos"))
        cmp["same_sign"] = (cmp.mean_in > 0) == (cmp.mean_oos > 0)
        cmp.to_csv(out / f"{tag}_in_vs_oos.csv", index=False)
        print("\nSame sign in and out of sample:", f"{cmp.same_sign.mean():.0%} of cells")

    print(f"\nSaved CSVs to {out.resolve()}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="outputs_8k", help="folder for CSV results")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("tags")

    s = sub.add_parser("screen")
    s.add_argument("--tags", nargs="+", required=True)
    s.add_argument("--start", default=L.STUDY_START)
    s.add_argument("--end", default=L.STUDY_END)
    s.add_argument("--max-events", type=int, default=None)

    t = sub.add_parser("study")
    t.add_argument("--tag", required=True)
    t.add_argument("--start", default=L.STUDY_START)
    t.add_argument("--end", default=L.STUDY_END)
    t.add_argument("--max-events", type=int, default=None)
    t.add_argument("--no-placebo", dest="placebo", action="store_false")
    t.add_argument("--no-oos", dest="oos", action="store_false")
    t.add_argument("--n-placebo", type=int, default=L.N_PLACEBO)

    args = p.parse_args()
    Path(args.out).mkdir(exist_ok=True)
    L.init_api()
    {"tags": cmd_tags, "screen": cmd_screen, "study": cmd_study}[args.cmd](args)


if __name__ == "__main__":
    main()
