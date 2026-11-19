"""The comparison's results from a folder of per-text score files: the metric table, flag rates, Litmus's own held-out
split, the cost table, `site/assets/results.json`, the Markdown tables, and every figure in both themes.

    python -m litmus_benchmarks results --scores BASELINES LITMUS --evaluation evaluation.json --costs costs.csv \\
        --model-version "VERSION" --work WORK --site ../site/assets --docs ../docs/images

Three choices live here, each in one place so a later source replaces it:

- Row filters: rows of a benchmark kept before any metric, from `catalog.csv`'s `row_filter` keys that name a
  score-file column (RAID extra keeps its Czech and German news), so a redefined benchmark is recomputed from the
  same score files; `ROW_FILTERS` stands in when no catalog is read.
- `TRAINED_ON`: which benchmarks each other detector saw in training (its training split, or for the predecessor the
  test texts themselves). Litmus's own come from `catalog.csv`'s `in_distribution` column, or from the command line
  for a build the catalog does not describe.
- `OWN_THRESHOLDS`: each detector's documented decision threshold, for flag rates at the threshold a user meets.
"""

from __future__ import annotations

import hashlib
import io
import json
import zlib
import numpy as np
import pandas as pd

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from sklearn.metrics import average_precision_score

from foundry.eval.bootstrap import stratified_bootstrap_indices
from .estimate import estimate_throughput
from .evaluate import METRICS, MIN_PER_CLASS, TASKS, bootstrap_metrics, evaluate_scores, length_bins, point_metrics


FLAG_RATE_BENCHMARKS = ("apt_eval", "apt_eval_human")
SCORE_COLUMNS = ("origin", "generator", "domain", "attack", "language")
# Used only when no catalog is read; the command takes each benchmark's filter from `catalog.csv`'s `row_filter`.
ROW_FILTERS: dict[str, dict[str, tuple[str, ...]]] = {"raid_extra": {"domain": ("czech", "german")}}
MAGE_SETS = frozenset({"mage", "mage_ood", "mage_ood_paraphrase"})
SEMEVAL_SETS = frozenset({"semeval2024_task8_a_mono", "semeval2024_task8_a_multi"})
TRAINED_ON: dict[str, frozenset[str]] = {
    # MAGE train, validation and test (whose rows include the two out-of-distribution sets), RAID train, SemEval-2024
    # Task 8 A train and dev, and COLING 2025's repack of M4GT (corpus/sources.csv, `in_first_corpus`).
    "ai-vs-human-checker": MAGE_SETS | SEMEVAL_SETS | {"raid_extra"},
    "Desklib v1.01": frozenset({"raid_extra"}),  # its card lists liamdugan/raid as training data
    "e5-small-lora": frozenset({"raid_extra"}),  # RAID train, per its card
    "ModernBERT RAID+MAGE (Drayson)": MAGE_SETS | {"raid_extra"},  # RAID and MAGE train, per its card
    "Binoculars": frozenset(),
    "Fast-DetectGPT": frozenset(),
    "Litmus": frozenset(),  # the interim build's corpus held every catalog benchmark out; `--litmus-trained-on` sets a later build's
}
OWN_THRESHOLDS = {
    "Litmus": 0.5,  # probability of not human above one half
    "ai-vs-human-checker": 0.024542788483370496,  # results.json of lhallee/ai-vs-human-checker at 234b5a2cca78
    "Desklib v1.01": 0.5,
    "e5-small-lora": 0.5,
    "ModernBERT RAID+MAGE (Drayson)": 0.5,
    "Binoculars": -0.8536432310785527,  # the authors' low-FPR threshold on the ratio, negated as the adapter negates it
}
COST_ALIASES = {"Litmus engine": "Litmus"}
BENCHMARK_NAMES = {
    "mage": "MAGE", "mage_ood": "MAGE unseen", "mage_ood_paraphrase": "MAGE paraphrased",
    "raid_extra": "RAID extra (cs, de)", "semeval2024_task8_a_mono": "SemEval-24 mono",
    "semeval2024_task8_a_multi": "SemEval-24 multi", "beemo": "Beemo", "mixset": "MixSet", "codet_m4": "CoDET-M4",
    "droid": "DroidCollection", "apt_eval": "APT-Eval polished", "apt_eval_human": "APT-Eval originals",
    "validation": "Litmus validation", "test": "Litmus test",
}
SITE_BENCHMARKS = {
    "raid_extra": {"id": "raid_extra", "name": "RAID extra, Czech and German news", "kind": "two_class",
                   "about": "RAID's labeled extra split without its code rows: news in Czech and German, every attack"},
    "validation": {"id": "validation", "name": "Litmus validation split", "kind": "two_class",
                   "about": "Held-out clusters of Litmus's own corpus, used to choose the model"},
}


def benchmark_name(benchmark: str) -> str:
    return BENCHMARK_NAMES.get(benchmark, benchmark)


def load_scores(folders: Sequence[Path]) -> pd.DataFrame:
    """Every `scores_<benchmark>__<detector>.csv` under the folders, one long table; a text scored twice by one
    detector (the same file in two folders) is an error."""
    paths = sorted(path for folder in folders for path in Path(folder).glob("scores_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no scores_*.csv under {list(folders)}")
    frames = [pd.read_csv(path, keep_default_na=False) for path in paths]
    pairs = [pair for frame in frames for pair in frame[["benchmark", "detector"]].drop_duplicates().itertuples(index=False)]
    if len(pairs) != len(set(pairs)):
        raise ValueError("two score files hold the same benchmark and detector; pass each once")
    return pd.concat(frames, ignore_index=True)


def filter_rows(scores: pd.DataFrame, filters: Mapping[str, Mapping[str, Sequence[str]]]) -> pd.DataFrame:
    """Rows kept by each benchmark's filter (column to allowed values); other benchmarks pass whole."""
    keep = np.ones(len(scores), dtype=bool)  # (n,)
    for benchmark, columns in filters.items():
        in_benchmark = (scores["benchmark"] == benchmark).to_numpy()  # (n,)
        allowed = np.ones(len(scores), dtype=bool)  # (n,)
        for column, values in columns.items():
            allowed &= scores[column].astype(str).isin([str(value) for value in values]).to_numpy()
        keep &= ~in_benchmark | allowed
    return scores[keep].reset_index(drop=True)


def catalog_filters(catalog: pd.DataFrame) -> dict[str, dict[str, tuple[str, ...]]]:
    """The catalog's `row_filter` keys that name a score-file column (a value or a list of values to keep); keys of
    raw columns (CoDET-M4's `split`) were applied when the benchmark was scored."""
    filters = {}
    for _, entry in catalog.iterrows():
        rule = json.loads(entry.get("row_filter") or "{}")
        columns = {column: tuple(str(value) for value in (wanted if isinstance(wanted, list) else [wanted]))
                   for column, wanted in rule.items() if column in SCORE_COLUMNS}
        if columns:
            filters[entry["name"]] = columns
    return filters


def parse_row_filters(texts: Sequence[str], base: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
                      ) -> dict[str, dict[str, tuple[str, ...]]]:
    """`BENCHMARK:COLUMN=V1,V2` overrides of `base` (`ROW_FILTERS` by default); `BENCHMARK:` alone removes a filter."""
    filters = {name: {column: tuple(values) for column, values in columns.items()}
               for name, columns in (ROW_FILTERS if base is None else base).items()}
    for text in texts:
        benchmark, _, rule = text.partition(":")
        if not rule:
            filters.pop(benchmark, None)
            continue
        column, _, values = rule.partition("=")
        filters.setdefault(benchmark, {})[column] = tuple(value.strip() for value in values.split(","))
    return filters


def trained_on(catalog: pd.DataFrame, litmus: Sequence[str] | None) -> dict[str, frozenset[str]]:
    """Benchmarks each detector trained on: `TRAINED_ON` for the other detectors; for Litmus, `litmus` when given
    (an empty list for a build that held every benchmark out), else the catalog's `in_distribution` column (`yes` where
    the corpus builder let the benchmark's training split in)."""
    mapping = dict(TRAINED_ON)
    if litmus is not None:
        mapping["Litmus"] = frozenset(litmus)
    elif "in_distribution" in catalog.columns:
        mapping["Litmus"] = frozenset(catalog.loc[catalog["in_distribution"].astype(str).str.strip() == "yes", "name"])
    return mapping


def in_distribution(mapping: Mapping[str, frozenset[str]], detector: str, benchmark: str) -> bool | None:
    """Whether the detector trained on the benchmark; None when no source says (a detector absent from the mapping)."""
    detector = COST_ALIASES.get(detector, detector)
    if detector not in mapping:
        return None
    return benchmark in mapping[detector]


def scores_digest(scores: pd.DataFrame, *settings: object) -> str:
    buffer = io.StringIO()
    scores.to_csv(buffer, index=False, lineterminator="\n")  # the platform's line ending would give Windows its own digest
    return hashlib.sha256((buffer.getvalue() + repr(settings)).encode()).hexdigest()[:16]


TEXT_KEYS = ["row", "origin", "generator", "domain", "attack", "words"]


def align_texts(scores: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Scores with a `text_set` column: the detectors of a benchmark that scored the same texts in the same order share
    one set (the largest keeps the benchmark's name, others get `<benchmark>#2`, ...), so metrics pair only detectors
    that read the same texts. A text one detector of a set could not score (an empty or non-finite score) leaves the
    set for every detector of it. Returns the notes of every split set and dropped text."""
    frames, notes = [], []
    for benchmark, block in scores.groupby("benchmark", sort=True):
        groups: dict[str, list[str]] = {}
        for detector, rows in block.groupby("detector", sort=True):
            groups.setdefault(hashlib.sha256(rows[TEXT_KEYS].to_csv(index=False).encode()).hexdigest(), []).append(detector)
        for index, detectors in enumerate(sorted(groups.values(), key=len, reverse=True)):
            part = block[block["detector"].isin(detectors)]
            part = part.assign(score=pd.to_numeric(part["score"], errors="coerce"), text_index=part.groupby("detector").cumcount())
            unscored = part.loc[~np.isfinite(part["score"]), "text_index"].unique()
            text_set = benchmark if index == 0 else f"{benchmark}#{index + 1}"
            frames.append(part[~part["text_index"].isin(unscored)].drop(columns="text_index").assign(text_set=text_set))
            if index or len(unscored):
                notes.append({"benchmark": benchmark, "text_set": text_set, "detectors": detectors,
                              "texts": int(part["text_index"].nunique() - len(unscored)), "unscored_texts_dropped": len(unscored),
                              "other_sample": bool(index)})
    return pd.concat(frames, ignore_index=True), notes


def cached_metrics(scores: pd.DataFrame, work: Path, *, replicates: int, strata_replicates: int,
                   workers: int = 1) -> pd.DataFrame:
    """`evaluate_scores` of the two-class benchmarks per text set, cached in `work` by the content of the scores; the
    table gains `text_set` beside `benchmark`. Text sets run in `workers` processes; every draw is seeded by its
    benchmark, task and stratum, so the table does not depend on how they are split."""
    two_class = scores[~scores["benchmark"].isin(FLAG_RATE_BENCHMARKS)]
    work.mkdir(parents=True, exist_ok=True)
    path = work / f"metrics_{scores_digest(two_class, replicates, strata_replicates)}.csv"
    if path.exists():
        return pd.read_csv(path, keep_default_na=False)
    by_set = two_class.drop(columns=["benchmark"]).rename(columns={"text_set": "benchmark"})
    parts = [block for _, block in by_set.groupby("benchmark", sort=True)]
    settings = {"replicates": replicates, "strata_replicates": strata_replicates}
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            tables = list(pool.map(partial(evaluate_scores, **settings), parts))
    else:
        tables = [evaluate_scores(part, **settings) for part in parts]
    metrics = pd.concat(tables, ignore_index=True)
    metrics = metrics.assign(text_set=metrics["benchmark"], benchmark=metrics["benchmark"].str.split("#").str[0])
    metrics.to_csv(path, index=False)
    return metrics


def scored_by_all(scores: pd.DataFrame, benchmarks: Sequence[str]) -> list[str]:
    """The benchmarks among `benchmarks` that every detector in `scores` scored."""
    detectors = set(scores["detector"])
    return [benchmark for benchmark in benchmarks if set(scores.loc[scores["benchmark"] == benchmark, "detector"]) == detectors]


@dataclass(frozen=True)
class BenchmarkSets:
    """The two-class benchmarks every detector scored (`every`), and of those the ones no detector trained on (`unseen`)."""

    every: list[str]
    unseen: list[str]


def benchmark_sets(scores: pd.DataFrame, mapping: Mapping[str, frozenset[str]]) -> BenchmarkSets:
    two_class = sorted(set(scores["benchmark"]) - set(FLAG_RATE_BENCHMARKS))
    every = scored_by_all(scores, two_class)
    detectors = sorted(set(scores["detector"]))
    unseen = [benchmark for benchmark in every
              if all(in_distribution(mapping, detector, benchmark) is False for detector in detectors)]
    return BenchmarkSets(every, unseen)


WIDE_KEYS = ("text_index", "origin", "attack")


def wide_scores(scores: pd.DataFrame, text_set: str) -> pd.DataFrame:
    """One row per AI or human text of a text set, one column per detector of it; the set's detectors scored the same
    texts in the same order (`align_texts`), so a text is its position in its detector's rows."""
    block = scores[scores["text_set"] == text_set]
    block = block.assign(text_index=block.groupby("detector").cumcount())
    block = block[block["origin"].isin(["ai", "human"])]
    return block.pivot(index=list(WIDE_KEYS), columns="detector", values="score").reset_index()


def text_sets(scores: pd.DataFrame, benchmark: str) -> list[str]:
    return sorted(scores.loc[scores["benchmark"] == benchmark, "text_set"].unique())


def mean_pr_auc(scores: pd.DataFrame, benchmarks: Sequence[str], *, replicates: int, seed: int = 0) -> pd.DataFrame:
    """Each detector's mean PR AUC (`ai_vs_human`) over `benchmarks`, with a bootstrap interval that resamples every
    benchmark's texts within class, the same draws for every detector that read the same texts."""
    detectors = sorted(scores["detector"].unique())
    totals = {detector: np.zeros(replicates) for detector in detectors}
    points: dict[str, list[float]] = {detector: [] for detector in detectors}
    for text_set in [name for benchmark in benchmarks for name in text_sets(scores, benchmark)]:
        wide = wide_scores(scores, text_set)
        labels = (wide["origin"] == "ai").to_numpy().astype(np.int64)  # (n,)
        rng = np.random.default_rng([seed, zlib.crc32(f"mean/{text_set}".encode())])
        draws = [stratified_bootstrap_indices(labels, rng) for _ in range(replicates)]  # replicates arrays of (n,)
        for detector in [column for column in wide.columns if column not in WIDE_KEYS]:
            values = wide[detector].to_numpy(dtype=np.float64)  # (n,)
            points[detector].append(average_precision_score(labels, values))
            totals[detector] += np.array([average_precision_score(labels[draw], values[draw]) for draw in draws])
    assert all(len(values) == len(benchmarks) for values in points.values()), "every detector scored every benchmark"
    rows = []
    for detector in detectors:
        low, high = np.percentile(totals[detector] / len(benchmarks), [2.5, 97.5])
        rows.append({"detector": detector, "value": float(np.mean(points[detector])), "low": float(low), "high": float(high),
                     "benchmarks": len(benchmarks)})
    return pd.DataFrame(rows)


def threshold_at_fpr(negatives: np.ndarray, fpr: float) -> float:
    """The score above which at most `fpr` of `negatives` lie (as `evaluate.tpr_at_fpr`)."""
    # negatives: (n,)
    ordered = np.sort(negatives)  # (n,)
    allowed = int(np.floor(fpr * len(ordered)))
    return float(ordered[len(ordered) - allowed - 1]) if allowed < len(ordered) else -np.inf


def flag_rates(scores: pd.DataFrame, pool: Sequence[str], *, replicates: int, seed: int = 0) -> pd.DataFrame:
    """Share of each one-origin set flagged at the detector's 1% and 5% FPR thresholds, set on the human texts of
    `pool` (the two-class benchmarks every detector scored), and at its own documented threshold, with bootstrap
    intervals that resample both the set and the human pool."""
    rows = []
    rng = np.random.default_rng(seed)
    for detector in sorted(scores["detector"].unique()):
        mine = scores[scores["detector"] == detector]
        human = mine.loc[mine["benchmark"].isin(pool) & (mine["origin"] == "human"), "score"].to_numpy()  # (h,)
        pool_draws = rng.integers(0, len(human), size=(replicates, len(human)))  # (r, h)
        for benchmark in FLAG_RATE_BENCHMARKS:
            values = mine.loc[mine["benchmark"] == benchmark, "score"].to_numpy()  # (t,)
            if not len(values):
                continue
            set_draws = rng.integers(0, len(values), size=(replicates, len(values)))  # (r, t)
            row = {"benchmark": benchmark, "detector": detector, "texts": len(values), "human_pool": len(human)}
            for name, fpr in (("flagged_at_fpr_1", 0.01), ("flagged_at_fpr_5", 0.05)):
                spread = [np.mean(values[set_draw] > threshold_at_fpr(human[pool_draw], fpr))
                          for set_draw, pool_draw in zip(set_draws, pool_draws, strict=True)]
                row[name] = float(np.mean(values > threshold_at_fpr(human, fpr)))
                row[f"{name}_low"], row[f"{name}_high"] = (float(value) for value in np.percentile(spread, [2.5, 97.5]))
            if detector in OWN_THRESHOLDS:
                flagged = values > OWN_THRESHOLDS[detector]  # (t,)
                spread = [flagged[set_draw].mean() for set_draw in set_draws]
                row["flagged_at_own_threshold"] = float(flagged.mean())
                row["flagged_at_own_threshold_low"], row["flagged_at_own_threshold_high"] = (
                    float(value) for value in np.percentile(spread, [2.5, 97.5]))
            rows.append(row)
    return pd.DataFrame(rows)


def own_split_rows(evaluation: Mapping[str, object], detector: str, *, replicates: int, seed: int = 0) -> pd.DataFrame:
    """Litmus's metrics on its own held-out split (`evaluation.json`'s passages, label 0 human, 1 ai, 2 mixed), every
    task, in the metric table's columns."""
    passages = evaluation["passages"]
    origins = np.array(["human", "ai", "mixed"])[np.array([passage["label"] for passage in passages])]  # (n,)
    probability = np.array([passage["probability"] for passage in passages], dtype=np.float64)  # (n,)
    records = []
    for task, (positive, negative) in TASKS.items():
        member = np.isin(origins, positive + negative)  # (n,)
        labels = np.isin(origins[member], positive).astype(np.int64)  # (m,)
        values = probability[member]  # (m,)
        rng = np.random.default_rng([seed, zlib.crc32(f"own/{task}".encode())])
        draws = [stratified_bootstrap_indices(labels, rng) for _ in range(replicates)]  # replicates arrays of (m,)
        point = point_metrics(labels, values)
        spread = bootstrap_metrics(labels, values, draws)
        for metric in METRICS:
            low, high = np.percentile(spread[metric], [2.5, 97.5])
            records.append({"benchmark": str(evaluation["split"]), "task": task, "stratum": "all", "subset": "all",
                            "detector": detector, "metric": metric, "value": point[metric], "low": float(low),
                            "high": float(high), "positives": int(labels.sum()), "negatives": int((1 - labels).sum()),
                            "replicates": replicates})
    return pd.DataFrame(records)


def attack_rows(scores: pd.DataFrame, benchmark: str, *, replicates: int, seed: int = 0) -> pd.DataFrame:
    """Each attack's AI texts against every human text of the benchmark (stratum `attack_vs_human`), every metric,
    `ai_vs_human`: RAID applies attacks to human text too, and a split with few human texts has too few per attack to
    compare within it."""
    records = []
    for text_set in text_sets(scores, benchmark):
        records += attack_records(wide_scores(scores, text_set), benchmark, text_set, replicates=replicates, seed=seed)
    return pd.DataFrame(records)


def attack_records(wide: pd.DataFrame, benchmark: str, text_set: str, *, replicates: int, seed: int) -> list[dict[str, object]]:
    records = []
    for attack in sorted(wide.loc[wide["origin"] == "ai", "attack"].unique()):
        member = ((wide["origin"] == "ai") & (wide["attack"] == attack) | (wide["origin"] == "human")).to_numpy()  # (n,)
        labels = (wide.loc[member, "origin"] == "ai").to_numpy().astype(np.int64)  # (m,)
        rng = np.random.default_rng([seed, zlib.crc32(f"{text_set}/attack/{attack}".encode())])
        draws = [stratified_bootstrap_indices(labels, rng) for _ in range(replicates)]  # replicates arrays of (m,)
        for detector in [column for column in wide.columns if column not in WIDE_KEYS]:
            values = wide.loc[member, detector].to_numpy(dtype=np.float64)  # (m,)
            point, spread = point_metrics(labels, values), bootstrap_metrics(labels, values, draws)
            for metric in METRICS:
                low, high = np.percentile(spread[metric], [2.5, 97.5])
                records.append({"benchmark": benchmark, "task": "ai_vs_human", "stratum": "attack_vs_human", "subset": attack,
                                "detector": detector, "metric": metric, "value": point[metric], "low": float(low),
                                "high": float(high), "positives": int(labels.sum()), "negatives": int((1 - labels).sum()),
                                "replicates": replicates, "text_set": text_set})
    return records


def length_rows(scores: pd.DataFrame, benchmarks: Sequence[str], *, replicates: int, seed: int = 0) -> pd.DataFrame:
    """Every metric by length bin (`evaluate.LENGTH_LABELS`, whitespace words) over the AI and human texts of
    `benchmarks` pooled, each detector on the texts it scored; AI and human texts of a bin are compared."""
    pooled = scores[scores["benchmark"].isin(benchmarks) & scores["origin"].isin(["ai", "human"])]
    pooled = pooled.assign(length=length_bins(pooled["words"].astype(int)))
    pooled = pooled[pooled["length"] != ""]
    records = []
    for (detector, length), block in pooled.groupby(["detector", "length"], sort=True):
        labels = (block["origin"] == "ai").to_numpy().astype(np.int64)  # (m,)
        if labels.sum() < MIN_PER_CLASS or (1 - labels).sum() < MIN_PER_CLASS:
            continue
        values = block["score"].to_numpy(dtype=np.float64)  # (m,)
        rng = np.random.default_rng([seed, zlib.crc32(f"length/{detector}/{length}".encode())])
        draws = [stratified_bootstrap_indices(labels, rng) for _ in range(replicates)]  # replicates arrays of (m,)
        point, spread = point_metrics(labels, values), bootstrap_metrics(labels, values, draws)
        for metric in METRICS:
            low, high = np.percentile(spread[metric], [2.5, 97.5])
            records.append({"benchmark": "pooled", "task": "ai_vs_human", "stratum": "length", "subset": length,
                            "detector": detector, "metric": metric, "value": point[metric], "low": float(low),
                            "high": float(high), "positives": int(labels.sum()), "negatives": int((1 - labels).sum()),
                            "replicates": replicates})
    return pd.DataFrame(records)


MEASURED_BASIS = {
    "Litmus engine": "measured: the shipped engine through its Python wheel, the whole reading",
    "Litmus": "measured: the PyTorch and NumPy reference path the scorer uses, not the shipped engine",
    "Binoculars": "measured on the first 20 texts", "Fast-DetectGPT": "measured on the first 20 texts",
}


def cost_table(costs: pd.DataFrame, architectures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measured rows and the estimated rows of leaderboard models, in one table, a row a detector; and the leaderboard
    models left without an estimate, with the reason."""
    assert not costs["detector"].duplicated().any(), "the cost table holds one measurement a detector"
    measured = costs.assign(measured=True, basis=costs["detector"].map(MEASURED_BASIS).fillna("measured"),
                            seconds_per_1k_words=costs["ms_per_1k_words"] / 1000)
    measured = measured.assign(detector=measured["detector"].replace({"Litmus": "Litmus (Python reference)"}))
    estimated = estimate_throughput(architectures, costs)
    usable = estimated[estimated["excluded"] == ""]
    estimated_rows = usable.assign(measured=False, machine=measured["machine"].iloc[0],
                                   seconds_per_1k_words=1000 / usable["words_per_second"],
                                   basis=[f"estimated from {row.anchor} x {row.ratio:.2f} (non-embedding parameters)"
                                          for row in usable.itertuples()])
    columns = ["detector", "tokens_per_second", "words_per_second", "seconds_per_1k_words", "measured", "basis", "machine",
               "texts", "peak_rss_mb"]
    table = pd.concat([measured, estimated_rows], ignore_index=True)
    return table.reindex(columns=columns), estimated[estimated["excluded"] != ""]


def site_rows(metrics: pd.DataFrame, flags: pd.DataFrame, mapping: Mapping[str, frozenset[str]]) -> list[dict[str, object]]:
    """`results.json` rows: one per benchmark, detector and task, every metric with its interval, then flag rates."""
    overall = metrics[metrics["stratum"] == "all"]
    rows = []
    for (benchmark, detector, task), block in overall.groupby(["benchmark", "detector", "task"], sort=True):
        values = block.set_index("metric")
        row: dict[str, object] = {"benchmark": benchmark, "detector": detector, "task": task}
        for metric in METRICS:
            row[metric] = round(float(values.loc[metric, "value"]), 4)
            row[f"{metric}_low"] = round(float(values.loc[metric, "low"]), 4)
            row[f"{metric}_high"] = round(float(values.loc[metric, "high"]), 4)
        row["positives"] = int(values["positives"].iloc[0])
        row["negatives"] = int(values["negatives"].iloc[0])
        row["in_distribution"] = in_distribution(mapping, detector, benchmark)
        text_set = values["text_set"].iloc[0] if "text_set" in values.columns else benchmark
        row["other_sample"] = isinstance(text_set, str) and text_set != benchmark
        rows.append(row)
    for record in flags.to_dict("records"):
        row = {"benchmark": record["benchmark"], "detector": record["detector"], "task": "flag_rate", "texts": record["texts"],
               "human_pool": record["human_pool"], "in_distribution": in_distribution(mapping, record["detector"], record["benchmark"])}
        row.update({key: round(float(value), 4) for key, value in record.items()
                    if key.startswith("flagged") and value == value})
        rows.append(row)
    return rows


def results_json(previous: Mapping[str, object], *, model_name: str, model_version: str, date: str,
                 rows: list[dict[str, object]], costs: pd.DataFrame, texts_per_benchmark: int, sets: BenchmarkSets,
                 fair: pd.DataFrame, every: pd.DataFrame, filters: Mapping[str, object]) -> dict[str, object]:
    """The site's `results.json` (schema 1, `site/README.md`), keeping the fields this command does not own (`release`,
    the benchmarks' wording) from the file it replaces; extra keys say how the comparison was made."""
    listed = [benchmark for benchmark in previous.get("benchmarks", [])]
    for benchmark_id, entry in SITE_BENCHMARKS.items():
        ids = [benchmark["id"] for benchmark in listed]
        if benchmark_id not in ids:  # beside the entry it belongs with: RAID extra after RAID, validation before test
            anchor = {"raid_extra": ("raid", 1), "validation": ("test", 0)}[benchmark_id]
            at = ids.index(anchor[0]) + anchor[1] if anchor[0] in ids else len(listed)
            listed.insert(at, entry)
    used = {row["benchmark"] for row in rows}
    benchmarks = [entry for entry in listed if entry["id"] in used or entry["id"] in ("test", "raid")]
    listed = {entry["id"]: entry for entry in listed}
    for benchmark_id in sorted(used - set(listed)):
        benchmarks.append({"id": benchmark_id, "name": benchmark_name(benchmark_id), "kind": "two_class", "about": ""})
    cost_rows = [{"detector": row.detector, "tokens_per_second": round(float(row.tokens_per_second)),
                  "words_per_second": round(float(row.words_per_second)),
                  "seconds_per_1k_words": round(float(row.seconds_per_1k_words), 4), "measured": bool(row.measured),
                  "machine": row.machine, "basis": row.basis} for row in costs.itertuples()]

    def means(table: pd.DataFrame) -> list[dict[str, object]]:
        return [{"detector": row.detector, "pr_auc": round(row.value, 4), "low": round(row.low, 4), "high": round(row.high, 4)}
                for row in table.itertuples()]

    return {
        "schema": 1,
        "model": {"name": model_name, "version": model_version, "date": date},
        "primary_task": "ai_vs_human",
        "texts_per_benchmark": texts_per_benchmark,
        "benchmarks": benchmarks,
        "rows": rows,
        "costs": cost_rows,
        "release": previous.get("release", {}),
        "comparison": {
            "row_filters": {name: {column: list(values) for column, values in columns.items()} for name, columns in filters.items()},
            "unseen_benchmarks": sets.unseen, "every_benchmark": sets.every,
            "mean_pr_auc_unseen": means(fair), "mean_pr_auc_every": means(every),
            "in_distribution": "true where the detector trained on the benchmark (its training split, or its test texts)",
        },
    }


TABLE_NAMES = {"ai-vs-human-checker": "ai-vs-human-checker (predecessor)", "Litmus": "**Litmus**"}


def interval(row: Mapping[str, object], metric: str) -> str:
    return f"{row[metric]:.3f} ({row[f'{metric}_low']:.3f} to {row[f'{metric}_high']:.3f})"


def markdown_tables(rows: list[dict[str, object]], costs: pd.DataFrame, excluded: pd.DataFrame, detectors_order: list[str]) -> str:
    """The README's and model card's tables: per benchmark, Litmus beside every rerun detector, all three metrics with
    intervals and the row counts, a dagger where the detector trained on the benchmark; flag rates; the cost table."""
    lines = []
    two_class = [row for row in rows if row["task"] == "ai_vs_human"]
    benchmarks = list(dict.fromkeys(row["benchmark"] for row in two_class))
    for benchmark in benchmarks:
        block = {row["detector"]: row for row in two_class if row["benchmark"] == benchmark}
        lines += [f"**{benchmark_name(benchmark)}**", "",
                  "| Detector | PR AUC | ROC AUC | Balanced MCC | TPR at 1% FPR | AI / human texts |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for detector in [name for name in detectors_order if name in block]:
            row = block[detector]
            mark = (" †" if row["in_distribution"] else "") + (" ‡" if row.get("other_sample") else "")
            lines.append(f"| {TABLE_NAMES.get(detector, detector)}{mark} | {interval(row, 'pr_auc')} | "
                         f"{interval(row, 'roc_auc')} | {interval(row, 'balanced_mcc')} | {interval(row, 'tpr_at_fpr_1')} | "
                         f"{row['positives']:,} / {row['negatives']:,} |")
        lines.append("")
    lines += ["† trained on this benchmark (its training split, or its test texts). ‡ read a different sample of the "
              "benchmark than the other detectors, so its intervals are not paired with theirs. Intervals are bootstrap 95%.", "",
              "**Flag rates on APT-Eval** (share flagged; thresholds at 1% and 5% FPR are set on the human texts of the "
              "benchmarks every detector scored)", "",
              "| Detector | Set | Texts | At 1% FPR | At 5% FPR | At its own threshold |", "| --- | --- | --- | --- | --- | --- |"]
    for row in sorted((row for row in rows if row["task"] == "flag_rate"),
                      key=lambda row: (row["benchmark"], detectors_order.index(row["detector"]) if row["detector"] in detectors_order else 99)):
        own = interval(row, "flagged_at_own_threshold") if "flagged_at_own_threshold" in row else "none documented"
        lines.append(f"| {TABLE_NAMES.get(row['detector'], row['detector'])} | {benchmark_name(row['benchmark'])} | {row['texts']:,} | "
                     f"{interval(row, 'flagged_at_fpr_1')} | {interval(row, 'flagged_at_fpr_5')} | {own} |")
    lines += ["", "**Single-core throughput**", "",
              "| Detector | Tokens a second | Seconds per 1,000 words | Basis |", "| --- | --- | --- | --- |"]
    for row in costs.sort_values("tokens_per_second", ascending=False).itertuples():
        lines.append(f"| {row.detector} | {row.tokens_per_second:,.0f} | {row.seconds_per_1k_words:.3f} | {row.basis} |")
    if len(excluded):
        lines += ["", "Left out of the estimates: " + "; ".join(f"{row.detector} ({row.excluded})" for row in excluded.itertuples()) + "."]
    return "\n".join(lines) + "\n"
