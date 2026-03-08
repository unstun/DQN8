#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "runs"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
CONDA_ENV = "ros2py310"
ALGORITHMS = (
    "mlp-dqn",
    "mlp-ddqn",
    "mlp-pddqn",
    "cnn-dqn",
    "cnn-ddqn",
    "cnn-pddqn",
)
PRETTY = {
    "mlp-dqn": "MLP-DQN",
    "mlp-ddqn": "MLP-DDQN",
    "mlp-pddqn": "MLP-PDDQN",
    "cnn-dqn": "CNN-DQN",
    "cnn-ddqn": "CNN-DDQN",
    "cnn-pddqn": "CNN-PDDQN",
}
PAIRWISE_CHECKS = (
    ("CNN-PDDQN", "CNN-DDQN"),
    ("CNN-DDQN", "CNN-DQN"),
    ("MLP-PDDQN", "MLP-DDQN"),
    ("MLP-DDQN", "MLP-DQN"),
    ("CNN-PDDQN", "MLP-PDDQN"),
    ("CNN-DDQN", "MLP-DDQN"),
    ("CNN-DQN", "MLP-DQN"),
)


@dataclass(frozen=True)
class ModelRef:
    source_type: str
    path: str
    epoch: int | None = None
    note: str = ""


@dataclass(frozen=True)
class Candidate:
    map_name: str
    name: str
    model_refs: dict[str, ModelRef]
    note: str = ""


@dataclass(frozen=True)
class ModeSpec:
    key: str
    profile: str
    out_dir: str
    metric_column: str
    higher_better: bool
    table_name: str
    quality_mode: bool


@dataclass(frozen=True)
class MapSpec:
    map_name: str
    env_name: str
    checkpoints_dir: str
    current_models_stage1: str
    current_models_validate: str
    validate_pairs_dir: str
    search_root: str
    validate_root: str


MAP_SPECS = {
    "forest": MapSpec(
        map_name="forest",
        env_name="forest_a",
        checkpoints_dir="runs/repro_20260226_v14b_1000ep/train_20260227_010647/checkpoints/forest_a",
        current_models_stage1="runs/codex_20260306_search/forest/current/models",
        current_models_validate="runs/codex_20260306_validate/forest/current/models",
        validate_pairs_dir="runs/codex_20260306_validate/forest/pairs",
        search_root="runs/codex_20260306_search/forest",
        validate_root="runs/codex_20260306_validate/forest",
    ),
    "realmap": MapSpec(
        map_name="realmap",
        env_name="realmap_a",
        checkpoints_dir="runs/repro_20260228_bug2fix_5000ep/train_20260228_052743/checkpoints/realmap_a",
        current_models_stage1="runs/codex_20260306_search/realmap/current/models",
        current_models_validate="runs/codex_20260306_validate/realmap/current/models",
        validate_pairs_dir="runs/codex_20260306_validate/realmap/pairs",
        search_root="runs/codex_20260306_search/realmap",
        validate_root="runs/codex_20260306_validate/realmap",
    ),
}


MODE_SPECS = {
    "forest": {
        "stage1": {
            "bk_long": ModeSpec("bk_long", "repro_20260306_codex_forest_stage1_bk_long", "runs/codex_20260306_search/forest/stage1_bk_long", "Success rate", True, "table2_kpis_mean.csv", False),
            "bk_short": ModeSpec("bk_short", "repro_20260306_codex_forest_stage1_bk_short", "runs/codex_20260306_search/forest/stage1_bk_short", "Success rate", True, "table2_kpis_mean.csv", False),
            "quality_long": ModeSpec("quality_long", "repro_20260306_codex_forest_stage1_quality_long", "runs/codex_20260306_search/forest/stage1_quality_long", "Composite score", False, "table2_kpis_mean_filtered.csv", True),
            "quality_short": ModeSpec("quality_short", "repro_20260306_codex_forest_stage1_quality_short", "runs/codex_20260306_search/forest/stage1_quality_short", "Composite score", False, "table2_kpis_mean_filtered.csv", True),
        },
        "validate": {
            "bk_long": ModeSpec("bk_long", "repro_20260306_codex_forest_validate_bk_long", "runs/codex_20260306_validate/forest/bk_long", "Success rate", True, "table2_kpis_mean.csv", False),
            "bk_short": ModeSpec("bk_short", "repro_20260306_codex_forest_validate_bk_short", "runs/codex_20260306_validate/forest/bk_short", "Success rate", True, "table2_kpis_mean.csv", False),
            "quality_long": ModeSpec("quality_long", "repro_20260306_codex_forest_validate_allsuc_long", "runs/codex_20260306_validate/forest/allsuc_long", "Composite score", False, "table2_kpis_mean_filtered.csv", True),
            "quality_short": ModeSpec("quality_short", "repro_20260306_codex_forest_validate_allsuc_short", "runs/codex_20260306_validate/forest/allsuc_short", "Composite score", False, "table2_kpis_mean_filtered.csv", True),
        },
    },
    "realmap": {
        "stage1": {
            "bk_long": ModeSpec("bk_long", "repro_20260306_codex_realmap_stage1_bk_long", "runs/codex_20260306_search/realmap/stage1_bk_long", "Success rate", True, "table2_kpis_mean.csv", False),
            "bk_short": ModeSpec("bk_short", "repro_20260306_codex_realmap_stage1_bk_short", "runs/codex_20260306_search/realmap/stage1_bk_short", "Success rate", True, "table2_kpis_mean.csv", False),
            "quality_long": ModeSpec("quality_long", "repro_20260306_codex_realmap_stage1_quality_long", "runs/codex_20260306_search/realmap/stage1_quality_long", "Composite score", False, "table2_kpis_mean_filtered.csv", True),
            "quality_short": ModeSpec("quality_short", "repro_20260306_codex_realmap_stage1_quality_short", "runs/codex_20260306_search/realmap/stage1_quality_short", "Composite score", False, "table2_kpis_mean_filtered.csv", True),
        },
        "validate": {
            "bk_long": ModeSpec("bk_long", "repro_20260306_codex_realmap_validate_bk_long", "runs/codex_20260306_validate/realmap/bk_long", "Success rate", True, "table2_kpis_mean.csv", False),
            "bk_short": ModeSpec("bk_short", "repro_20260306_codex_realmap_validate_bk_short", "runs/codex_20260306_validate/realmap/bk_short", "Success rate", True, "table2_kpis_mean.csv", False),
            "quality_long": ModeSpec("quality_long", "repro_20260306_codex_realmap_validate_allsuc_long", "runs/codex_20260306_validate/realmap/allsuc_long", "Composite score", False, "table2_kpis_mean_filtered.csv", True),
            "quality_short": ModeSpec("quality_short", "repro_20260306_codex_realmap_validate_allsuc_short", "runs/codex_20260306_validate/realmap/allsuc_short", "Composite score", False, "table2_kpis_mean_filtered.csv", True),
        },
    },
}


EXISTING_RESULT_SETS = {
    "forest_f0": {
        "map": "forest",
        "candidate": "F0",
        "modes": {
            "bk_long": "runs/snapshot_20260304_4modes_v3/results/mode1_bk_long",
            "bk_short": "runs/snapshot_20260304_4modes_v3/results/mode2_bk_short",
            "quality_long": "runs/snapshot_20260304_4modes_v3/results/mode3_allsuc_long",
            "quality_short": "runs/snapshot_20260304_4modes_v3/results/mode4_allsuc_short",
        },
    },
    "forest_f1": {
        "map": "forest",
        "candidate": "F1",
        "modes": {
            "bk_long": "runs/repro_20260226_v14b_1000ep/train_20260227_010647/infer/20260306_013752",
            "bk_short": "runs/repro_20260226_v14b_1000ep/train_20260227_010647/infer/20260306_015901",
            "quality_long": "runs/repro_20260226_v14b_1000ep/train_20260227_010647/infer/20260304_171032",
            "quality_short": "runs/repro_20260226_v14b_1000ep/train_20260227_010647/infer/20260304_192308",
        },
    },
    "realmap_r0": {
        "map": "realmap",
        "candidate": "R0",
        "modes": {
            "bk_long": "runs/snapshot_20260305_realmap_4modes_v1/results/mode1/20260305_012620",
            "bk_short": "runs/snapshot_20260305_realmap_4modes_v1/results/mode2/20260305_014915",
            "quality_long": "runs/snapshot_20260305_realmap_4modes_v1/results/mode3/20260305_015930",
            "quality_short": "runs/snapshot_20260305_realmap_4modes_v1/results/mode4/20260305_022223",
        },
    },
}


def project_path(path_text: str) -> Path:
    return PROJECT_ROOT / path_text


def checkpoint_ref(map_name: str, algo: str, epoch: int) -> ModelRef:
    spec = MAP_SPECS[map_name]
    path = project_path(spec.checkpoints_dir) / f"{algo}_ep{epoch:05d}.pt"
    return ModelRef(source_type="checkpoint", path=str(path.relative_to(PROJECT_ROOT)), epoch=epoch)


def file_ref(path_text: str, note: str) -> ModelRef:
    return ModelRef(source_type="file", path=path_text, epoch=None, note=note)


def forest_candidates() -> dict[str, Candidate]:
    snapshot_root = Path("runs/snapshot_20260304_4modes_v3/models")
    candidates: dict[str, Candidate] = {
        "F0": Candidate(
            map_name="forest",
            name="F0",
            note="snapshot_20260304_4modes_v3 统一模型集",
            model_refs={algo: file_ref(str(snapshot_root / f"{algo}.pt"), "snapshot_20260304_4modes_v3") for algo in ALGORITHMS},
        ),
        "F1": Candidate(
            map_name="forest",
            name="F1",
            note="Forest all-490 canonical 候选",
            model_refs={algo: checkpoint_ref("forest", algo, 490) for algo in ALGORITHMS},
        ),
    }
    for cnn_dqn_epoch in (300, 490):
        for cnn_ddqn_epoch in (300, 490):
            name = f"Fgrid_cnn-dqn{cnn_dqn_epoch}_cnn-ddqn{cnn_ddqn_epoch}"
            model_refs = {algo: checkpoint_ref("forest", algo, 490) for algo in ALGORITHMS}
            model_refs["cnn-dqn"] = checkpoint_ref("forest", "cnn-dqn", cnn_dqn_epoch)
            model_refs["cnn-ddqn"] = checkpoint_ref("forest", "cnn-ddqn", cnn_ddqn_epoch)
            candidates[name] = Candidate(
                map_name="forest",
                name=name,
                note="Forest canonical 微调网格",
                model_refs=model_refs,
            )
    return candidates


def realmap_candidates() -> dict[str, Candidate]:
    candidates = {
        "R0": Candidate(
            map_name="realmap",
            name="R0",
            note="Realmap 既有统一候选",
            model_refs={
                "mlp-dqn": checkpoint_ref("realmap", "mlp-dqn", 450),
                "mlp-ddqn": checkpoint_ref("realmap", "mlp-ddqn", 450),
                "mlp-pddqn": checkpoint_ref("realmap", "mlp-pddqn", 5000),
                "cnn-dqn": checkpoint_ref("realmap", "cnn-dqn", 1000),
                "cnn-ddqn": checkpoint_ref("realmap", "cnn-ddqn", 1000),
                "cnn-pddqn": checkpoint_ref("realmap", "cnn-pddqn", 3000),
            },
        ),
        "R1": Candidate(
            map_name="realmap",
            name="R1",
            note="Realmap 质量友好候选",
            model_refs={
                "mlp-dqn": checkpoint_ref("realmap", "mlp-dqn", 450),
                "mlp-ddqn": checkpoint_ref("realmap", "mlp-ddqn", 450),
                "mlp-pddqn": checkpoint_ref("realmap", "mlp-pddqn", 5000),
                "cnn-dqn": checkpoint_ref("realmap", "cnn-dqn", 1000),
                "cnn-ddqn": checkpoint_ref("realmap", "cnn-ddqn", 1000),
                "cnn-pddqn": checkpoint_ref("realmap", "cnn-pddqn", 1000),
            },
        ),
    }
    return candidates


CANDIDATES = {
    "forest": forest_candidates(),
    "realmap": realmap_candidates(),
}


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_workspace() -> None:
    for spec in MAP_SPECS.values():
        (project_path(spec.current_models_stage1) / spec.env_name).mkdir(parents=True, exist_ok=True)
        (project_path(spec.current_models_validate) / spec.env_name).mkdir(parents=True, exist_ok=True)
        project_path(spec.validate_pairs_dir).mkdir(parents=True, exist_ok=True)
        project_path(spec.search_root).mkdir(parents=True, exist_ok=True)
        project_path(spec.validate_root).mkdir(parents=True, exist_ok=True)


def candidate_names_for_map(map_name: str) -> list[str]:
    return sorted(CANDIDATES[map_name])


def resolve_candidate(map_name: str, candidate_name: str) -> Candidate:
    try:
        return CANDIDATES[map_name][candidate_name]
    except KeyError as exc:
        raise SystemExit(f"Unknown candidate for {map_name}: {candidate_name}") from exc


def stage_models_dir(map_name: str, stage: str) -> Path:
    spec = MAP_SPECS[map_name]
    models_dir = spec.current_models_stage1 if stage == "stage1" else spec.current_models_validate
    return project_path(models_dir) / spec.env_name


def materialize_candidate(candidate: Candidate, stage: str) -> dict[str, object]:
    target_dir = stage_models_dir(candidate.map_name, stage)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "candidate": candidate.name,
        "map": candidate.map_name,
        "stage": stage,
        "note": candidate.note,
        "traceable": all(ref.source_type == "checkpoint" for ref in candidate.model_refs.values()),
        "models": {},
    }
    for algo in ALGORITHMS:
        ref = candidate.model_refs[algo]
        source_path = project_path(ref.path)
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source model: {source_path}")
        target_path = target_dir / f"{algo}.pt"
        shutil.copy2(source_path, target_path)
        manifest["models"][algo] = {
            "source_type": ref.source_type,
            "source_path": str(source_path.relative_to(PROJECT_ROOT)),
            "epoch": ref.epoch,
            "note": ref.note,
            "target_path": str(target_path.relative_to(PROJECT_ROOT)),
            "md5": md5sum(target_path),
        }
    manifest_path = target_dir.parent / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def latest_timestamped_dir(experiment_dir: Path) -> Path | None:
    if not experiment_dir.exists():
        return None
    candidates = [path for path in experiment_dir.iterdir() if path.is_dir() and path.name[:8].isdigit()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item.name)[-1]


def load_metric_table(table_path: Path) -> dict[str, dict[str, float | int | str]]:
    rows: dict[str, dict[str, float | int | str]] = {}
    with table_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            algo_name = row.get("Algorithm name", "").strip()
            if not algo_name:
                continue
            payload: dict[str, float | int | str] = {"Algorithm name": algo_name}
            for key, value in row.items():
                if key == "Algorithm name" or value is None or value == "":
                    continue
                try:
                    number = float(value)
                except ValueError:
                    payload[key] = value
                else:
                    if float(number).is_integer():
                        payload[key] = int(number)
                    else:
                        payload[key] = number
            rows[algo_name] = payload
    return rows


def rank_map(metric_values: dict[str, float], higher_better: bool) -> dict[str, float]:
    ordered = sorted(metric_values.items(), key=lambda item: (-item[1], item[0]) if higher_better else (item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank_value = (index + 1 + end) / 2.0
        for offset in range(index, end):
            ranks[ordered[offset][0]] = rank_value
        index = end
    return ranks


def summarize_result_set(map_name: str, mode_dirs: dict[str, Path], *, partial: bool = False) -> dict[str, object]:
    mode_specs = MODE_SPECS[map_name]["stage1"]
    mode_payloads: dict[str, dict[str, object]] = {}
    pairwise_details: list[dict[str, object]] = []
    rank_accumulator = {pretty: [] for pretty in PRETTY.values()}
    quality_best = True
    quality_margins: list[float] = []
    total_checks = 0
    passed_checks = 0

    for mode_name, result_dir in mode_dirs.items():
        mode_spec = mode_specs[mode_name]
        table_path = result_dir / mode_spec.table_name
        if not table_path.exists():
            raise FileNotFoundError(f"Missing result table: {table_path}")
        rows = load_metric_table(table_path)
        metric_values = {pretty: float(rows[pretty][mode_spec.metric_column]) for pretty in PRETTY.values()}
        ranks = rank_map(metric_values, mode_spec.higher_better)
        for algo_name, rank_value in ranks.items():
            rank_accumulator[algo_name].append(rank_value)
        if mode_spec.quality_mode:
            drl_values = {pretty: metric_values[pretty] for pretty in PRETTY.values()}
            ordered = sorted(drl_values.items(), key=lambda item: item[1])
            best_name, best_value = ordered[0]
            second_value = ordered[1][1]
            quality_best = quality_best and best_name == "CNN-PDDQN"
            quality_margins.append(float(second_value - best_value))
        for left_name, right_name in PAIRWISE_CHECKS:
            left_value = metric_values[left_name]
            right_value = metric_values[right_name]
            passed = left_value >= right_value if mode_spec.higher_better else left_value <= right_value
            total_checks += 1
            passed_checks += int(passed)
            pairwise_details.append(
                {
                    "mode": mode_name,
                    "left": left_name,
                    "right": right_name,
                    "left_value": left_value,
                    "right_value": right_value,
                    "passed": passed,
                }
            )
        filtered_runs = None
        if mode_spec.quality_mode:
            filtered_runs = int(rows["CNN-PDDQN"].get("Filtered runs", 0))
        mode_payloads[mode_name] = {
            "result_dir": str(result_dir.relative_to(PROJECT_ROOT)),
            "table": str(table_path.relative_to(PROJECT_ROOT)),
            "metric_column": mode_spec.metric_column,
            "higher_better": mode_spec.higher_better,
            "metrics": metric_values,
            "filtered_runs": filtered_runs,
        }

    average_ranks = {algo_name: sum(values) / len(values) for algo_name, values in rank_accumulator.items() if values}
    cnn_order = average_ranks["CNN-PDDQN"] < average_ranks["CNN-DDQN"] < average_ranks["CNN-DQN"]
    mlp_order = average_ranks["MLP-PDDQN"] < average_ranks["MLP-DDQN"] < average_ranks["MLP-DQN"]
    cnn_family = (average_ranks["CNN-PDDQN"] + average_ranks["CNN-DDQN"] + average_ranks["CNN-DQN"]) / 3.0
    mlp_family = (average_ranks["MLP-PDDQN"] + average_ranks["MLP-DDQN"] + average_ranks["MLP-DQN"]) / 3.0
    family_order = cnn_family < mlp_family
    pairwise_threshold = 23 if not partial and total_checks == 28 else max(1, int(total_checks * 0.75))
    story_pass = quality_best and cnn_order and mlp_order and family_order and passed_checks >= pairwise_threshold

    return {
        "map": map_name,
        "partial": partial,
        "modes": mode_payloads,
        "pairwise_pass_count": passed_checks,
        "pairwise_total": total_checks,
        "pairwise_threshold": pairwise_threshold,
        "pairwise_checks": pairwise_details,
        "quality_best_drl": quality_best,
        "quality_margin_sum": sum(quality_margins),
        "average_ranks": average_ranks,
        "cnn_rank_order_ok": cnn_order,
        "mlp_rank_order_ok": mlp_order,
        "cnn_family_better_than_mlp": family_order,
        "story_pass": story_pass,
    }


def compare_story_scores(left_score: dict[str, object], right_score: dict[str, object], *, left_traceable: bool, right_traceable: bool) -> int:
    left_key = (
        int(bool(left_score["story_pass"])),
        int(left_traceable),
        int(left_score["pairwise_pass_count"]),
        float(left_score["quality_margin_sum"]),
    )
    right_key = (
        int(bool(right_score["story_pass"])),
        int(right_traceable),
        int(right_score["pairwise_pass_count"]),
        float(right_score["quality_margin_sum"]),
    )
    if left_key > right_key:
        return -1
    if right_key > left_key:
        return 1
    return 0


def score_existing_preset(preset_name: str) -> dict[str, object]:
    preset = EXISTING_RESULT_SETS[preset_name]
    mode_dirs = {mode_name: project_path(path_text) for mode_name, path_text in preset["modes"].items()}
    score = summarize_result_set(str(preset["map"]), mode_dirs)
    score["candidate"] = preset["candidate"]
    score["preset"] = preset_name
    return score


def write_markdown_summary(output_path: Path, title: str, payload: dict[str, object]) -> None:
    lines = [f"# {title}", ""]
    lines.append(f"- story_pass: {payload['story_pass']}")
    lines.append(f"- quality_best_drl: {payload['quality_best_drl']}")
    lines.append(f"- pairwise: {payload['pairwise_pass_count']}/{payload['pairwise_total']}")
    lines.append(f"- cnn_rank_order_ok: {payload['cnn_rank_order_ok']}")
    lines.append(f"- mlp_rank_order_ok: {payload['mlp_rank_order_ok']}")
    lines.append(f"- cnn_family_better_than_mlp: {payload['cnn_family_better_than_mlp']}")
    lines.append("")
    lines.append("## Modes")
    lines.append("")
    for mode_name, mode_payload in payload["modes"].items():
        lines.append(f"### {mode_name}")
        lines.append(f"- result_dir: `{mode_payload['result_dir']}`")
        lines.append(f"- table: `{mode_payload['table']}`")
        if mode_payload["filtered_runs"] is not None:
            lines.append(f"- filtered_runs: {mode_payload['filtered_runs']}")
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def config_path(profile: str) -> Path:
    return CONFIGS_ROOT / f"{profile}.json"


def run_infer_profile(profile: str) -> subprocess.CompletedProcess[str]:
    command = [
        "conda",
        "run",
        "--cwd",
        str(PROJECT_ROOT),
        "-n",
        CONDA_ENV,
        "python",
        "infer.py",
        "--profile",
        profile,
    ]
    return subprocess.run(command, check=False, text=True, capture_output=True)


def run_candidate(candidate: Candidate, stage: str, modes: Iterable[str], execute: bool) -> dict[str, object]:
    ensure_workspace()
    manifest = materialize_candidate(candidate, stage)
    selected_modes = list(modes)
    mode_specs = MODE_SPECS[candidate.map_name][stage]
    planned = []
    result_dirs: dict[str, Path] = {}
    for mode_name in selected_modes:
        mode_spec = mode_specs[mode_name]
        profile_file = config_path(mode_spec.profile)
        if not profile_file.exists():
            raise FileNotFoundError(f"Missing config profile: {profile_file}")
        planned.append({
            "mode": mode_name,
            "profile": mode_spec.profile,
            "config": str(profile_file.relative_to(PROJECT_ROOT)),
            "out_dir": mode_spec.out_dir,
        })
        if execute:
            completed = run_infer_profile(mode_spec.profile)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"infer failed for {mode_spec.profile}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
                )
            latest_dir = latest_timestamped_dir(project_path(mode_spec.out_dir))
            if latest_dir is None:
                raise FileNotFoundError(f"No infer output produced under {mode_spec.out_dir}")
            result_dirs[mode_name] = latest_dir
    payload: dict[str, object] = {
        "candidate": candidate.name,
        "map": candidate.map_name,
        "stage": stage,
        "execute": execute,
        "manifest": manifest,
        "planned_runs": planned,
    }
    if execute:
        payload["score"] = summarize_result_set(candidate.map_name, result_dirs, partial=set(selected_modes) != {"bk_long", "bk_short", "quality_long", "quality_short"})
    return payload


def search_plan_payload() -> dict[str, object]:
    return {
        "forest": {
            "baselines": ["F0", "F1"],
            "fallback_grid": [
                "Fgrid_cnn-dqn300_cnn-ddqn300",
                "Fgrid_cnn-dqn300_cnn-ddqn490",
                "Fgrid_cnn-dqn490_cnn-ddqn300",
                "Fgrid_cnn-dqn490_cnn-ddqn490",
            ],
            "mode_order": ["bk_long", "bk_short", "quality_long", "quality_short"],
        },
        "realmap": {
            "baselines": ["R0", "R1"],
            "sweep_order": [
                {"algo": "cnn-ddqn", "epochs": [500, 1000, 1500], "modes": ["bk_short", "quality_long", "quality_short"]},
                {"algo": "cnn-dqn", "epochs": [1000, 1500, 3000], "modes": ["bk_short", "quality_long", "quality_short"]},
                {"algo": "cnn-pddqn", "epochs": [1000, 2500, 3000], "modes": ["bk_long", "bk_short", "quality_long", "quality_short"]},
                {"algo": "mlp-pddqn", "epochs": [4000, 5000], "modes": ["bk_short", "quality_long"]},
            ],
        },
    }


def write_search_plan_files() -> None:
    payload = search_plan_payload()
    json_path = RUNS_ROOT / "codex_20260306_search" / "search_plan.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    markdown_path = RUNS_ROOT / "codex_20260306_search" / "search_plan.md"
    lines = ["# Codex 2026-03-06 Search Plan", "", "## Forest", ""]
    lines.append("- baselines: `F0`, `F1`")
    lines.append("- fallback grid: `cnn-dqn` in {300,490}, `cnn-ddqn` in {300,490}")
    lines.append("")
    lines.append("## Realmap")
    lines.append("")
    lines.append("- baselines: `R0`, `R1`")
    lines.append("- sweep order: `cnn-ddqn` -> `cnn-dqn` -> `cnn-pddqn` -> `mlp-pddqn`")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_prepare() -> None:
    ensure_workspace()
    write_search_plan_files()
    output_dir = RUNS_ROOT / "codex_20260306_search" / "existing_scores"
    output_dir.mkdir(parents=True, exist_ok=True)
    for preset_name in EXISTING_RESULT_SETS:
        score = score_existing_preset(preset_name)
        json_path = output_dir / f"{preset_name}.json"
        json_path.write_text(json.dumps(score, indent=2, sort_keys=True), encoding="utf-8")
        md_path = output_dir / f"{preset_name}.md"
        write_markdown_summary(md_path, f"Existing score: {preset_name}", score)
    print(f"Prepared workspace under {output_dir.relative_to(PROJECT_ROOT)}")


def command_materialize(args: argparse.Namespace) -> None:
    candidate = resolve_candidate(args.map, args.candidate)
    manifest = materialize_candidate(candidate, args.stage)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def command_score_existing(args: argparse.Namespace) -> None:
    score = score_existing_preset(args.preset)
    print(json.dumps(score, indent=2, sort_keys=True))


def command_run_candidate(args: argparse.Namespace) -> None:
    candidate = resolve_candidate(args.map, args.candidate)
    modes = args.modes or ["bk_long", "bk_short", "quality_long", "quality_short"]
    payload = run_candidate(candidate, args.stage, modes, args.execute)
    output_dir = RUNS_ROOT / "codex_20260306_search" / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.map}_{args.candidate}_{args.stage}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {output_path.relative_to(PROJECT_ROOT)}")
    if not args.execute:
        print(json.dumps(payload, indent=2, sort_keys=True))


def command_list(args: argparse.Namespace) -> None:
    if args.map:
        names = candidate_names_for_map(args.map)
        print("\n".join(names))
        return
    for map_name in sorted(CANDIDATES):
        print(f"[{map_name}]")
        for name in candidate_names_for_map(map_name):
            print(name)


def command_search_plan() -> None:
    payload = search_plan_payload()
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint story lock helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Create isolated workspace and score existing baselines")
    prepare_parser.set_defaults(func=lambda args: command_prepare())

    list_parser = subparsers.add_parser("list-candidates", help="List candidates")
    list_parser.add_argument("--map", choices=sorted(CANDIDATES), default=None)
    list_parser.set_defaults(func=command_list)

    materialize_parser = subparsers.add_parser("materialize", help="Copy a candidate into isolated current/models")
    materialize_parser.add_argument("--map", required=True, choices=sorted(CANDIDATES))
    materialize_parser.add_argument("--candidate", required=True)
    materialize_parser.add_argument("--stage", required=True, choices=("stage1", "validate"))
    materialize_parser.set_defaults(func=command_materialize)

    score_parser = subparsers.add_parser("score-existing", help="Score an existing preset result set")
    score_parser.add_argument("--preset", required=True, choices=sorted(EXISTING_RESULT_SETS))
    score_parser.set_defaults(func=command_score_existing)

    run_parser = subparsers.add_parser("run-candidate", help="Materialize a candidate and optionally execute infer")
    run_parser.add_argument("--map", required=True, choices=sorted(CANDIDATES))
    run_parser.add_argument("--candidate", required=True)
    run_parser.add_argument("--stage", required=True, choices=("stage1", "validate"))
    run_parser.add_argument("--mode", dest="modes", action="append", choices=("bk_long", "bk_short", "quality_long", "quality_short"))
    run_parser.add_argument("--execute", action="store_true")
    run_parser.set_defaults(func=command_run_candidate)

    plan_parser = subparsers.add_parser("search-plan", help="Print the staged search order")
    plan_parser.set_defaults(func=lambda args: command_search_plan())
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
