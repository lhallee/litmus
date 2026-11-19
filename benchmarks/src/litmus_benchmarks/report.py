"""The one command that turns score files, an `evaluation.json` and cost rows into the site's results, the Markdown
tables and every figure; rerun it whenever a score file changes.

    python -m litmus_benchmarks results --scores BASELINES LITMUS --evaluation evaluation.json \\
        --costs costs.csv [costs.csv ...] --model-version VERSION --work WORK \\
        --site ../site/assets --docs ../docs/images

The work folder receives the cached metric table, the flag rates, the cost table and `results.md`; the site folder
receives `results.json` and `figures/<name>_<theme>.png`; the docs folder receives copies named
`benchmarks_<name>_<theme>.png`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pandas as pd

from pathlib import Path

from .results import (FLAG_RATE_BENCHMARKS, align_texts, attack_rows, benchmark_sets, cached_metrics, catalog_filters, cost_table, filter_rows, flag_rates,
                      length_rows, load_scores, markdown_tables, mean_pr_auc, own_split_rows, parse_row_filters,
                      results_json, site_rows, trained_on)


def add_arguments(parser: argparse.ArgumentParser, here: Path) -> None:
    parser.add_argument("--scores", type=Path, nargs="+", required=True, help="folders of scores_<benchmark>__<detector>.csv")
    parser.add_argument("--evaluation", type=Path, help="Litmus's evaluation.json on its own held-out split")
    parser.add_argument("--costs", type=Path, nargs="+", required=True, help="CSV files of the cost command's rows; a detector in several keeps its row in the last")
    parser.add_argument("--architectures", type=Path, default=here / "architectures.csv")
    parser.add_argument("--catalog", type=Path, default=here / "catalog.csv")
    parser.add_argument("--published", type=Path, default=here / "published.csv")
    parser.add_argument("--model-name", default="Litmus")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--date", default=datetime.datetime.now(datetime.UTC).date().isoformat())
    parser.add_argument("--litmus-trained-on", nargs="*", help="benchmarks this Litmus build trained on; given with no "
                        "names, none (a build that held every benchmark out); omitted, the catalog's in_distribution column")
    parser.add_argument("--row-filter", nargs="*", default=[], help="BENCHMARK:COLUMN=V1,V2 overrides of the catalog's "
                        "row_filter; BENCHMARK: removes one")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--strata-replicates", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1, help="processes for the metric table")
    parser.add_argument("--metrics", type=Path, help="a metric table already computed from the same scores (--metrics-only)")
    parser.add_argument("--metrics-only", action="store_true", help="write the metric table to WORK and stop")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True, help="the site's assets folder")
    parser.add_argument("--docs", type=Path, help="the README's image folder")


def build(args: argparse.Namespace) -> list[Path]:
    """Every output of the comparison; returns the paths written."""
    args.work.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv(args.catalog, keep_default_na=False)
    mapping = trained_on(catalog, args.litmus_trained_on)
    filters = parse_row_filters(args.row_filter, catalog_filters(catalog))
    scores, notes = align_texts(filter_rows(load_scores(args.scores), filters))
    if args.metrics:
        metrics = pd.read_csv(args.metrics, keep_default_na=False)
        computed = set(zip(metrics["text_set"], metrics["detector"], strict=True))
        expected = {(text_set, detector) for text_set, detector in zip(scores["text_set"], scores["detector"], strict=True)
                    if text_set.split("#")[0] not in FLAG_RATE_BENCHMARKS}
        assert computed == expected, f"--metrics was computed from other scores: {sorted(computed ^ expected)[:5]}"
    else:
        metrics = cached_metrics(scores, args.work, replicates=args.replicates, strata_replicates=args.strata_replicates,
                                 workers=args.workers)
    if args.metrics_only:
        return sorted(args.work.glob("metrics_*.csv"))
    from .figures import detector_order, draw_site_figures  # matplotlib only here; the launch image runs without it

    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8")) if args.evaluation else None
    own = own_split_rows(evaluation, args.model_name, replicates=args.replicates) if evaluation else metrics.iloc[:0]
    sets = benchmark_sets(scores, mapping)
    fair = mean_pr_auc(scores, sets.unseen, replicates=args.replicates)
    every = mean_pr_auc(scores, sets.every, replicates=args.replicates)
    flags = flag_rates(scores, sets.every, replicates=args.replicates)
    attacks = attack_rows(scores, "raid_extra", replicates=args.strata_replicates)
    lengths = length_rows(scores, sets.every, replicates=args.strata_replicates)
    # A detector measured in several files keeps its last row: the release's own cost file, given last, replaces the
    # row an earlier engine left in the baselines' files.
    measured = pd.concat([pd.read_csv(path) for path in args.costs], ignore_index=True).drop_duplicates("detector", keep="last")
    costs, excluded = cost_table(measured, pd.read_csv(args.architectures))
    rows = site_rows(pd.concat([own, metrics], ignore_index=True), flags, mapping)
    target = args.site / "results.json"
    previous = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    texts = int(scores.groupby(["benchmark", "detector"]).size().max())
    document = results_json(previous, model_name=args.model_name, model_version=args.model_version, date=args.date,
                            rows=rows, costs=costs, texts_per_benchmark=texts, sets=sets, fair=fair, every=every,
                            filters=filters)
    document["comparison"]["text_sets"] = notes
    document["comparison"]["estimates_left_out"] = [{"detector": row.detector, "reason": row.excluded}
                                                    for row in excluded.itertuples()]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    order = detector_order(scores["detector"].unique())
    (args.work / "results.md").write_text(markdown_tables(rows, costs, excluded, order), encoding="utf-8", newline="\n")
    for name, table in (("flags", flags), ("costs", costs), ("attacks", attacks), ("lengths", lengths),
                        ("mean_pr_auc_unseen", fair), ("mean_pr_auc_every", every), ("own_split", own)):
        table.to_csv(args.work / f"{name}.csv", index=False)
    published = pd.read_csv(args.published, keep_default_na=False)
    figures = draw_site_figures({"metrics": metrics, "attacks": attacks, "lengths": lengths, "published": published,
                                 "scores": scores, "costs": costs, "fair": fair, "every": every, "mapping": mapping,
                                 "evaluation": evaluation}, args.site / "figures", args.docs)
    return [target, args.work / "results.md", *figures]
