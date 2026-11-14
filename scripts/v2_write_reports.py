"""Publish bilingual overviews, offline data and report from completed v2 outputs."""

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from cruxrnn.config import load_config
from cruxrnn.engine import cell_directory, make_jobs
from cruxrnn.reproducibility import file_hash, utc_now, write_json


def table(frame, columns, names=None):
    names = names or {}
    lines = [
        "| " + " | ".join(names.get(column, column) for column in columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for column, value in zip(columns, row):
            if isinstance(value, (float, np.floating)):
                values.append(
                    "undefined"
                    if not math.isfinite(value)
                    else f"{value:.4g}"
                    if column.startswith("p_")
                    else f"{value:.3f}"
                )
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    config = load_config("configs/v2/main.json")
    target = Path("results/v2")
    verification = json.loads((target / "verification.json").read_text())
    if (
        verification["missing_cells"]
        or verification["completed_cells"] != 1260
        or verification["main_models"] != 2760
    ):
        raise RuntimeError(
            "Reports require all actual planned v2 cells; never substitute target counts"
        )
    summary = pd.read_csv(target / "main_summary.csv")
    runs = pd.read_csv(target / "run_level.csv")
    effects = pd.read_csv(target / "variant_effects.csv")
    contrasts = pd.read_csv(target / "variant_contrasts.csv")
    granularity = pd.read_csv(target / "granularity_effects.csv")
    interactions = json.loads((target / "interactions.json").read_text())
    records = [
        json.loads(path.read_text())
        for path in sorted((target / "execution_records").glob("*.json"))
    ]
    producing = [
        record
        for record in records
        if any(item["status"] == "completed" for item in record["successes"])
    ]
    wall = sum(record["wall_seconds"] for record in producing)
    original = summary[summary.distance.isin([1, 2, 4, 7])]
    zero = summary[(summary.ci_low_pp <= 0) & (summary.ci_high_pp >= 0)].distance.tolist()
    zero_text = (
        "Intervals include zero at k=" + ", ".join(map(str, zero)) + "."
        if zero
        else "All seven pointwise intervals exclude zero."
    )
    primary_sig = original[original.p_holm_primary_family < 0.05].distance.tolist()
    sig_text = (
        "Primary distance tests passing Holm correction: "
        + (", ".join(map(str, primary_sig)) if primary_sig else "none")
        + "."
    )
    slope = interactions["original_four"]
    all_slope = interactions["all_distances"]
    slope_text = f"Original-four slope: {slope['slope_pp_per_hamming']:.3f} pp per Hamming unit, 95% CI [{slope['ci_low']:.3f}, {slope['ci_high']:.3f}], Holm p={slope['p_slope_holm']:.4g}."
    names = {
        "distance": "k",
        "n": "Pairs",
        "blocked_pct": "Blocked %",
        "interleaved_pct": "IP %",
        "difference_pp": "IP−blocked pp",
        "ci_low_pp": "CI low",
        "ci_high_pp": "CI high",
        "sd_difference_pp": "SD pp",
        "median_pp": "Median pp",
        "p_holm_primary_family": "Primary Holm p",
    }
    primary_table = table(
        original,
        [
            "distance",
            "n",
            "blocked_pct",
            "interleaved_pct",
            "difference_pp",
            "ci_low_pp",
            "ci_high_pp",
        ],
        names,
    )
    full_table = table(
        summary,
        [
            "distance",
            "n",
            "blocked_pct",
            "interleaved_pct",
            "difference_pp",
            "median_pp",
            "sd_difference_pp",
            "ci_low_pp",
            "ci_high_pp",
            "paired_dz",
            "p_holm_primary_family",
        ],
        names,
    )
    variants_table = table(
        effects,
        [
            "architecture",
            "hidden_size",
            "distance",
            "n",
            "difference_pp",
            "ci_low_pp",
            "ci_high_pp",
            "paired_dz",
        ],
    )
    contrasts_table = table(
        contrasts,
        [
            "family",
            "architecture",
            "hidden_size",
            "distance",
            "difference_pp",
            "ci_low_pp",
            "ci_high_pp",
            "p_holm_family",
        ],
    )
    granularity_table = table(
        granularity,
        ["distance", "schedule", "n", "difference_pp", "ci_low_pp", "ci_high_pp", "p_holm_family"],
    )
    main_runs = runs[
        (runs.architecture == "rnn")
        & (runs.hidden_size == 7)
        & runs.schedule.isin(["blocked", "interleaved"])
    ]
    secondary = (
        main_runs.groupby(["distance", "schedule"])[
            [
                "clean_accuracy",
                "sequence_accuracy",
                "cross_entropy",
                "generalization_clean",
                "generalization_noisy",
                "forgetting_noisy",
            ]
        ]
        .mean()
        .reset_index()
    )
    for column in (
        "clean_accuracy",
        "sequence_accuracy",
        "generalization_clean",
        "generalization_noisy",
        "forgetting_noisy",
    ):
        secondary[column] *= 100
    secondary_table = table(secondary, list(secondary.columns))
    seq = main_runs.pivot(
        index=["run_id", "distance"], columns="schedule", values="sequence_accuracy"
    ).reset_index()
    seq["difference"] = seq.interleaved - seq.blocked
    lower = [int(k) for k, part in seq.groupby("distance") if part.difference.mean() < 0]
    contradiction = (
        "IP has lower mean exact-sequence accuracy at k="
        + ",".join(map(str, lower))
        + ", despite any action-accuracy advantage."
        if lower
        else "The exact-sequence outcome should still be read separately from action accuracy."
    )
    negative = effects[effects.difference_pp < 0][
        ["architecture", "hidden_size", "distance", "difference_pp"]
    ]
    negative_text = (
        table(negative, list(negative.columns))
        if len(negative)
        else "No variant has a negative mean raw paired difference; interval uncertainty still applies."
    )
    matched_k7 = effects[effects.distance == 7]
    capacity_values = "; ".join(
        f"h={int(row.hidden_size)}: {row.difference_pp:.2f} pp"
        for row in matched_k7[matched_k7.architecture == "rnn"].itertuples()
    )
    architecture_values = "; ".join(
        f"{row.architecture.upper()} h={int(row.hidden_size)}: {row.difference_pp:.2f} pp"
        for row in matched_k7[matched_k7.architecture != "rnn"].itertuples()
    )
    sensitivity_text = f"In the matched 20-ID k=7 subset, RNN capacity effects were {capacity_values}. Larger capacity did not remove the schedule gap under this fixed protocol. Gated-model effects were {architecture_values}; several low-distance intervals include zero. These are exploratory protocol comparisons, not optimized architecture rankings."
    noise = pd.read_csv(target / "noise_summary.csv")
    noise = noise[(noise.architecture == "rnn") & (noise.hidden_size == 7)]
    noise_pairs = noise.pivot(
        index=["distance", "noise_sd"], columns="schedule", values="mean"
    ).reset_index()
    noise_pairs["difference_pp"] = 100 * (noise_pairs.interleaved - noise_pairs.blocked)
    noise_at_one = "; ".join(
        f"k={int(row.distance)}: {row.difference_pp:.2f} pp"
        for row in noise_pairs[noise_pairs.noise_sd == 1].itertuples()
    )
    noise_text = (
        "At held-out SD=1 the raw mean differences were "
        + noise_at_one
        + ". These robustness means are descriptive; no additional corrected significance claim is made."
    )
    branch = "v2-research-upgrade"
    header = f"""<div align="center">
<img src="docs/assets/cruxrnn-logo.svg" width="112" alt="CruxRNN climbing and recurrence logo">
<h1>CruxRNN</h1>
<p><strong>Exploring how sequence similarity shapes learning, memory, and interference in recurrent neural networks.</strong></p>
<a href="https://github.com/ericqin0816/crux-rnn/actions/workflows/ci.yml"><img src="https://github.com/ericqin0816/crux-rnn/actions/workflows/ci.yml/badge.svg?branch={branch}" alt="CI"></a>
<img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" alt="Python 3.8 and newer">
<img src="https://img.shields.io/badge/PyTorch-1.13%2B-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 1.13 and newer">
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT license scope in NOTICE"></a>
<p><a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a></p>
</div>
"""
    quick = """```bash
git clone https://github.com/ericqin0816/crux-rnn.git
cd crux-rnn
git switch v2-research-upgrade
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -p "test_v2*.py" -v
cruxrnn demo --output outputs/my-demo
```
"""
    structure = """```text
src/cruxrnn/          typed v2 models, data, schedules, training, statistics, CLI
src/motor_rnn/        frozen v1 numerical implementation
configs/v2/          committed main, pilot and smoke recipes
results/v2/          actual seed-level results, figures and provenance
tests/               historical and expanded scientific controls
docs/                methods, statistics, reproduction and owner guide
docs/demo/           offline interactive HTML/SVG results explorer
reports/             separate v1 and v2 research reports
```
"""
    readme = (
        header
        + f"""
**v2 review candidate.** The upgrade is on a separate branch; v1.0 remains
preserved. Merge and final v2 release await owner review.

CruxRNN is an independent computational replication and controlled extension
of [Song, Kim & Kim (2026)](https://doi.org/10.1038/s41598-026-40162-w).
It asks how similarity between seven-action patterns changes the relative
effects of concentrated blocked practice and shuffled interleaved practice.
Climbing motivated the question; these are artificial-network simulations,
with no human participants or climbing measurements.

![Measured v2 accuracy across all seven distances](results/v2/figures/main_accuracy.png)

## Measured results

**100 pairs at each k=1–7: 1400 main schedule models.** Smaller matched
extensions bring the completed scientific total to **2760 main models**;
D-adapted clones are counted separately. No v1, smoke or pilot data are pooled.
The original-distance primary endpoint is fresh-noisy action accuracy before D:

{primary_table}

{zero_text} {sig_text} {slope_text}
{contradiction} Statistics concern this model/generator, not human mechanisms.
See the [full report](reports/v2_research_report.md) for all seven conditions,
secondary outcomes, adjusted tests, effect sizes and sensitivity analyses.

## Quick start

Use an activated virtual environment; the tiny demo runs on CPU and is not a
scientific result. [Installation and exact versions](docs/reproducibility.md).

{quick}
The v2 suite supports modern PyTorch. The complete historical test suite and
bitwise research reproduction use the separately documented Python 3.8.10 /
PyTorch 1.13.1 CPU environment. Both CI jobs pass on their stated environments.

## Method and extensions

Within each pair, weights and noisy examples are identical; only presentation
order changes. All three pairwise sequence distances equal k. Pretrain once,
then train each schedule for 750 SGD updates and test on independently noisy
held-out cues. Copies and action positions are not independent subjects.

- RNN / GRU / LSTM, with parameter counts and approximate matching.
- Fully blocked, 1000/100-example blocks, trial cycling and shuffled IP.
- RNN capacities h=7,16,32,64; fresh cue-noise robustness.
- A/B/C learning dynamics, exact-sequence scores and matched D interference.

[Methods](docs/methodology.md) · [Experiments](docs/experiments.md) ·
[Statistical plan](docs/statistical_analysis.md) · [Code audit](docs/code_audit.md)

## Explore and reproduce

Download the repository and open [docs/demo/index.html](docs/demo/index.html).
The self-contained viewer needs no server or frontend install. It separates
illustrative schedules from actual results; unmeasured conditions stay missing.
No site or final v2 release is deployed before review.

```bash
cruxrnn run --config configs/v2/main.json --workers 4
cruxrnn verify --config configs/v2/main.json
cruxrnn analyze --config configs/v2/main.json
cruxrnn figures --config configs/v2/main.json
```

[Download compact seed-level data](https://raw.githubusercontent.com/ericqin0816/crux-rnn/{branch}/results/v2/run_level.csv)
· [Tables, figures and checksum indexes](results/v2/)
· [Owner study guide](docs/researcher_guide.md)
· [Preserved v1 release and raw archive](https://github.com/ericqin0816/crux-rnn/releases/tag/v1.0-final-experiment)

Full v2 checkpoints/arrays are supplied as a local archive with an integrity
receipt, pending an owner-approved release. They are excluded from Git.
Measured scientific execution: {wall:.2f} seconds ({wall / 60:.2f} minutes),
excluding analysis/figure/report preparation. See provenance for exact times.

## Repository

{structure}
## Limits and acknowledgment

The full cue contains its targets. Ordinal labels, unmatched action
frequencies/repeats, fixed budgets, observed D distances and sign-flip
assumptions limit generality. Equal-budget architecture comparisons do not
estimate each architecture's optimized potential. No biological mechanism or
climbing-performance effect is demonstrated. [Full limitations](docs/limitations.md).

Read [CITATION.cff](CITATION.cff), acknowledge the source paper and
[pinned source repository](https://github.com/YJ-0000/RNN_during_RP_and_IP),
and cite the exact version used. The proposed MIT scope covers independently
authored CruxRNN code/assets; third-party rights remain separate in
[NOTICE.md](NOTICE.md). Development was AI-assisted and requires independent
owner review. [Contributing](CONTRIBUTING.md) · [Changelog](CHANGELOG.md) ·
[Original v1 overview](README.v1.md).
"""
    )
    Path("README.md").write_text(readme, encoding="utf-8")
    chinese_table = table(
        original,
        [
            "distance",
            "n",
            "blocked_pct",
            "interleaved_pct",
            "difference_pp",
            "ci_low_pp",
            "ci_high_pp",
        ],
        {
            "distance": "汉明距离 k",
            "n": "配对次数",
            "blocked_pct": "集中练习 %",
            "interleaved_pct": "交错练习 %",
            "difference_pp": "平均差值（百分点）",
            "ci_low_pp": "区间下限",
            "ci_high_pp": "区间上限",
        },
    )
    chinese_zero = (
        "以下 k 的区间包含零，效应方向仍不确定：" + "、".join(map(str, zero)) + "。"
        if zero
        else "七个条件的逐点区间均不包含零；逐点区间不等同于多重比较校正后的结论。"
    )
    chinese_sig = (
        "原始距离条件中，通过 Holm 校正的检验为："
        + ("、".join(map(str, primary_sig)) if primary_sig else "无")
        + "。"
    )
    chinese_lower = (
        "在 k="
        + "、".join(map(str, lower))
        + " 时，交错练习的平均完整序列准确率更低，因此动作准确率的优势不能概括为所有指标都更好。"
        if lower
        else "完整序列准确率与动作准确率应分别解读。"
    )
    chinese_header = header.replace(
        "Exploring how sequence similarity shapes learning, memory, and interference in recurrent neural networks.",
        "探索序列相似性如何塑造循环神经网络中的学习、记忆与干扰。",
    )
    chinese_structure = structure.replace(
        "typed v2 models, data, schedules, training, statistics, CLI",
        "带类型标注的 v2 模型、数据、练习顺序、训练、统计与命令行",
    )
    chinese_structure = chinese_structure.replace(
        "frozen v1 numerical implementation", "冻结的 v1 数值实现"
    ).replace("committed main, pilot and smoke recipes", "已提交的主实验、试运行与冒烟测试配置")
    chinese_structure = chinese_structure.replace(
        "actual seed-level results, figures and provenance", "逐次实测结果、图与溯源记录"
    ).replace("historical and expanded scientific controls", "历史与扩展科学验证测试")
    chinese_structure = (
        chinese_structure.replace(
            "methods, statistics, reproduction and owner guide", "方法、统计、复现与所有者指南"
        )
        .replace("offline interactive HTML/SVG results explorer", "离线交互式 HTML/SVG 结果查看器")
        .replace("separate v1 and v2 research reports", "独立的 v1 与 v2 研究报告")
    )
    zh = (
        chinese_header
        + f"""
**v2 审阅候选版。** 升级位于独立分支，v1.0 保持可访问；合并与正式发布 v2 均须等待项目所有者审阅。

CruxRNN 是对 [Song、Kim 和 Kim（2026）](https://doi.org/10.1038/s41598-026-40162-w)
的独立计算复现与受控扩展。研究问题是：七个动作组成的序列越相似或越不相似，
集中练习与交错练习对循环神经网络的影响有何变化？攀岩中的重复动作与相近路线
启发了这个问题，但本项目没有人类被试、脑部记录或攀岩测量数据。

![v2 在全部七个汉明距离下的实测准确率](results/v2/figures/main_accuracy.png)

## 实测结果

**k=1–7 每个条件完成 100 次配对，共 1400 个主实验模型。** 加上使用较小样本的
配对扩展，科学实验共完成 **2760 个主训练模型**；学习 D 后的模型副本另行计数。
v1、冒烟测试和试运行的数据均未并入最终推断。原始四个距离条件的主要指标是
学习 D 之前、加入独立新噪声的最终动作准确率：

{chinese_table}

{chinese_zero} {chinese_sig}
原始四个距离的斜率为每单位汉明距离 {slope["slope_pp_per_hamming"]:.3f} 个百分点，
95% 区间 [{slope["ci_low"]:.3f}, {slope["ci_high"]:.3f}]，Holm 校正 p={slope["p_slope_holm"]:.4g}。
{chinese_lower} 统计结论只针对本模型与数据生成规则，不代表人类神经机制。
全部七个条件、次要指标、效应量、校正检验与敏感性分析见[完整报告](reports/v2_research_report.md)。

## 快速开始

先激活独立 Python 虚拟环境。轻量演示使用 CPU，不能当作科学结论。
详细安装步骤与精确版本见[复现指南](docs/reproducibility.md)。

{quick}
v2 测试支持较新的 PyTorch。完整历史测试与逐位数值复现使用单独记录的
Python 3.8.10 / PyTorch 1.13.1 CPU 环境。两个 CI 任务均在各自声明的环境中通过。

## 方法与扩展

每次配对中，模型初始权重和含噪训练样本完全相同，主要操纵变量是呈现顺序。
A、B、C 三对序列的汉明距离均等于 k。每个单元只预训练一次，各练习条件完成
750 次 SGD 更新，随后用独立噪声的保留数据评估。含噪副本与动作位置不算独立样本。

- 比较 RNN、GRU 与 LSTM，报告参数量，并提供近似参数匹配。
- 比较集中练习、每 1000/100 个样本交替、逐试次循环与随机交错练习。
- 检查 h=7、16、32、64 的模型容量，以及新输入噪声下的稳健性。
- 记录 A/B/C 学习过程、完整序列准确率，以及学习相同 D 序列后的干扰。

[方法](docs/methodology.md) · [实验](docs/experiments.md) ·
[统计方案](docs/statistical_analysis.md) · [代码审计](docs/code_audit.md)

## 交互查看与复现

下载仓库后打开 [docs/demo/index.html](docs/demo/index.html)。这个独立 HTML 页面
不需要服务器或前端依赖；练习顺序示意图与实测结果明确分开，未测条件不会编造数值。
审阅前不部署网站，也不发布正式 v2 版本。

```bash
cruxrnn run --config configs/v2/main.json --workers 4
cruxrnn verify --config configs/v2/main.json
cruxrnn analyze --config configs/v2/main.json
cruxrnn figures --config configs/v2/main.json
```

[下载精简的逐次结果数据](https://raw.githubusercontent.com/ericqin0816/crux-rnn/{branch}/results/v2/run_level.csv)
· [表格、图与校验索引](results/v2/)
· [项目所有者学习指南](docs/researcher_guide.md)
· [保留的 v1 版本与原始数据包](https://github.com/ericqin0816/crux-rnn/releases/tag/v1.0-final-experiment)

v2 的完整权重与数组以附有完整性记录的本地数据包交付，等待所有者批准发布；
这些大文件不进入 Git。科学实验实测运行时间为 {wall:.2f} 秒（{wall / 60:.2f} 分钟），
不含统计分析、制图与报告整理；精确时间见运行溯源记录。

## 仓库结构

{chinese_structure}
## 局限与致谢

初始提示包含全部目标，因而这是完整提示的数值解码任务。动作的序数编码、未匹配的
动作频率与重复模式、固定训练预算、未操纵的 D 距离，以及符号翻转检验的假设，
均限制结论的适用范围。相同预算下的架构比较不代表各架构经过充分调参后的上限。
本项目不能证明人脑机制或攀岩表现的改善。[完整局限](docs/limitations.md)。

引用时请使用 [CITATION.cff](CITATION.cff)，注明实际使用版本，并致谢原论文及
[固定提交的原始仓库](https://github.com/YJ-0000/RNN_during_RP_and_IP)。拟采用的 MIT
许可仅覆盖独立编写的 CruxRNN 代码与原创视觉素材；第三方权利范围见
[NOTICE.md](NOTICE.md)。开发与文档使用了 AI 辅助，须由项目所有者独立审阅。
[贡献指南](CONTRIBUTING.md) · [更新记录](CHANGELOG.md) · [v1 原始说明](README.v1.md)。
"""
    )
    Path("README.zh-CN.md").write_text(zh, encoding="utf-8")
    report = f"""# CruxRNN v2 research report

**Computational replication and extension; review candidate, not a published paper or human-subject study.**

## Abstract

We examined whether sequence similarity moderates the effect of practice
schedule in complete-cue recurrent decoding. The main study completed 100
pairs at every nonzero seven-position Hamming distance. Architecture,
capacity and granularity extensions reused matched 20-ID reference subsets.
All {verification["main_models"]} planned scientific main models and the
same number of D-adapted clones completed in {wall:.2f} seconds wall time.
{zero_text} {slope_text} {contradiction}
The result concerns a fixed artificial protocol and cannot establish human
neural mechanisms or climbing-performance effects.

## Background and question

Song et al. (2026) studied blocked and interleaved practice in a minimal Elman
network. CruxRNN independently reproduced the audited computation, then
controlled pairwise sequence distance. Climbing supplies motivation, not a
validation dataset. v2 asks whether the schedule-by-similarity relationship
persists with more seeds and under architecture, capacity, granularity,
interference and cue-noise changes.

## Hypotheses and prospective timing

H1 tests a paired schedule effect at original k=1,2,4,7. H2 tests moderation
by distance, with linear and categorical interaction analyses. The versioned
[plan](../docs/experiment_plan_v2.md) was committed at `9c64a49` before final
data generation. Prior v1 outcomes were seen; this is not a claim that earlier
outcome exposure can be undone. Smoke, pilot, parity and custom/demo runs are
separate and excluded.

## Methods

The complete seven-action cue is shifted from 0–3 targets to 1–4 inputs with
Gaussian SD .1 noise. Exactly k positions contain three distinct A/B/C actions;
all other positions agree. Pretraining, main, unseen and D bases are unique
and disjoint. One shared pretrained network per variant is cloned into
schedules using identical examples, targets and initial state hashes.
SGD .02, zero momentum, batch 20, 1000 pretraining examples, 15,000 main
examples/750 updates and 2000 D examples/100 updates are fixed.

All six blocked orders occur 16 or 17 times per distance. Additional schedules
change granularity while preserving exposure/update counts. GRU/LSTM h=7 and
h=3 and RNN h=16/32/64 use the same budgets/families as the matched references.
Approximate matching compares RNN7=179 parameters to GRU3=159 and LSTM3=195;
same-hidden GRU7/LSTM7 have 403/515. D and robustness reuse trained models.
[Full methods](../docs/methodology.md).

## Main results

All seven conditions contain the planned 100 complete pairs. Accuracy is the
fresh-noisy final action score before D; differences/intervals are pp.
Raw/standardized bootstrap intervals are pointwise. Full precision, SDs and
sensitivity columns are retained in `results/v2/main_summary.csv`.

{full_table}

{zero_text} {sig_text} Do not equate unadjusted pointwise intervals with Holm
familywise decisions or confuse a small supported effect with practical value.

## Interaction and sensitivity

{slope_text}
Original-four categorical interaction: adjusted p={slope["p_categorical_holm"]:.4g}.
All-seven slope: {all_slope["slope_pp_per_hamming"]:.3f} pp/unit,
95% CI [{all_slope["ci_low"]:.3f}, {all_slope["ci_high"]:.3f}], secondary-family
Holm p={all_slope["p_slope_holm"]:.4g}. Original-four leave-one-ID-out slopes
range [{slope["leave_one_out_slope_min"]:.3f}, {slope["leave_one_out_slope_max"]:.3f}].
Paired t-tests and 10% trimmed means are retained as sensitivities, not
replacements for the prespecified endpoint. The categorical plot remains
necessary because a slope need not describe every intermediate distance.

## Exploratory architecture and capacity results

The following raw variant effects use the same first 20 family IDs at each
original distance, including the reused RNN7 reference. Intervals are
descriptive pointwise intervals; the next table tests corrected matched
differences in schedule effect against RNN7.

{variants_table}

Difference-in-differences (variant effect minus reference effect), with Holm
correction within architecture or capacity families:

{contrasts_table}

Negative mean raw variant effects, retained without exclusion:

{negative_text}

{sensitivity_text}
These equal-budget comparisons do not estimate optimized architecture
performance. Parameter counts, pretraining behavior and gate dynamics differ.

## Exploratory granularity

Each alternative is compared with fully blocked practice on the same 20 IDs
and examples. Four alternatives × four distances form its Holm family.

{granularity_table}

## Secondary outcomes and contradictions

Accuracy/generalization columns below are percent, forgetting is pp, and
cross-entropy is unscaled loss. Clean/noisy, action/exact-sequence outcomes
answer different questions.

{secondary_table}

{contradiction} Learning-speed attainment counts and reached-only medians are
in `thresholds.csv`; nonattainment is censored. A/B/C and position groups,
shared/differing scores, before/after-D metrics and all noise levels remain
in the raw/group tables. Shared k=7 accuracy is NaN with count zero. Per-seed
per-sequence forgetting distributions are in `forgetting_distributions.csv`.
Raw D drops must be interpreted with baseline/floor effects and observed D
distances; they do not identify biological consolidation.

{noise_text}

## Statistical analysis

Independent IDs are the units. Ten thousand paired percentile bootstrap
resamples give raw and paired-dz intervals. Sign-flip tests use the stated
symmetry assumption and plus-one Monte Carlo correction. Primary Holm family:
four original-distance effects plus linear/categorical interactions (six).
The all-seven secondary family has nine. Interaction resamples/sign flips
retain complete ID rows. Uncertainty concerns this model/generator.
[Detailed assumptions](../docs/statistical_analysis.md).

## Reproducibility, failures and engineering validation

Completed cells: {verification["completed_cells"]}/{verification["expected_cells"]}.
Main models: {verification["main_models"]}; D clones: {verification["D_adapted_clones"]}.
Recorded final execution failures: {verification["failure_records"]}; exclusions:
{verification["excluded_cells"]}. Producing Git commit(s):
{", ".join("`" + commit + "`" for commit in verification["producing_git_commits"])}.
Namespace: `{config.namespace}`. Config SHA-256:
`{file_hash("configs/v2/main.json")}`. Phase start/end UTC:
`{min(record["started_at_utc"] for record in producing)}` /
`{max(record["finished_at_utc"] for record in producing)}`.
Wall time {wall:.3f} s is not the sum of parallel cell times and excludes
analysis/figures/reporting. Full environment/freeze and execution records are
saved alongside exact seeds, sequence sets, schedule IDs, metrics, checkpoints
and SHA-256 indexes.

All 38 recorded-environment tests passed. Eight full-budget v1 models reproduced
every final weight and all 750 losses bitwise. Smoke: 64 cells / 152 main
models; full-budget pilot: 32 / 76; quick-start demo: 16 / 38. A fresh modern
clone passed 21 v2 tests and its separate 38-model demo. These diagnostics
are not pooled. GitHub Actions checks the recorded and modern environments.

Prototype integration exposed a Windows index-dtype issue and old pip lacked
editable-build support. Both were fixed before final data, without changing
the experimental design. CUDA remains untested on this CPU-only setup.

## Discussion and limits

Read effect magnitude, intervals, corrected tests and secondary outcomes
together. A difference within this controlled network does not establish a
general advantage of one curriculum. Full target cues, ordinal coding,
unmatched action frequencies/repeats, fixed optimization, unequal capacities,
observed D similarity, smaller exploratory samples and prior outcome exposure
limit conclusions. [Full limitations](../docs/limitations.md).

The primary mean gap grows from {summary.iloc[0].difference_pp:.2f} pp at k=1
to {summary.iloc[-1].difference_pp:.2f} pp at k=7. Both schedules have lower
absolute action accuracy at k=7 than at k=1; blocked accuracy declines more.
The k=1 paired median is {summary.iloc[0].median_pp:.2f} pp, so a positive mean
does not imply an advantage for every seed. In this dataset IP's mean raw D
forgetting is greater at every distance, despite its higher pre-D accuracy;
baseline/floor effects and final retained scores constrain that interpretation.

## Future directions

Independently replicate with new namespaces; match frequencies/repeats; test
categorical relabeling, one-hot/partial cues and justified architecture-specific
optimization; manipulate D similarity; and compare predictions with suitable
behavioral measurements. These require new plans rather than post hoc claims.

## Data, figures and references

Compact outputs: `results/v2/`; full local arrays/checkpoints/bootstrap arrays:
`outputs/v2/main/`. Twelve original PNG/SVG figure types are under
`results/v2/figures/`; learning bands are ±SE, paired intervals are bootstrap
95% CIs. The offline viewer reads only exported actual tables and recorded
families; schedule illustrations are separately labeled.

Song, Y., Kim, H., & Kim, T. (2026). A minimal recurrent neural network models
the robustness of interleaved practice on motor sequence learning. Scientific
Reports, 16, 10068. https://doi.org/10.1038/s41598-026-40162-w .
Pinned source: `31f58bb6de90d1fffcb752c0b97afb7df51479a7`.
Original source code/figures are excluded; CruxRNN visuals are original.

AI-assisted development and documentation are disclosed. Owner review is
required before merge/final v2 release; this report is not publication approval.
"""
    Path("reports/v2_research_report.md").write_text(report, encoding="utf-8")
    export_demo(config, target, verification)
    changelog = f"""# Changelog

## 2.0.0rc1 — review candidate, 2026-11-14

- Audited the complete v1 scientific pipeline and established a passing baseline.
- Added a typed v2 package, exact seven-distance data, configurable SGD,
  five schedules, RNN/GRU/LSTM, resume guards, device support and installed CLI.
- Prespecified and completed {verification["main_models"]} scientific main models,
  with separate D clones; retained every seed and all secondary outcomes.
- Added paired effect sizes, bootstrap CIs, sign-flip tests, interaction,
  Holm correction and sensitivity analyses.
- Added 21 v2 tests (38 total in the reference environment), two green CI jobs,
  fresh-install checks, original figures/logo, bilingual overview, offline viewer
  and an owner study guide. Proposed MIT scope and AI assistance are explicit.
- No merge or final v2 release is performed before owner review.

## 1.0 — 2026-10-13 (America/Chicago)

The preserved `v1.0-final-experiment` release completed 120 pairs / 240 main
models with a frozen config, contemporary provenance and separate exploratory
history. Tag commit: `c1fdcadaca0ace46cf590bbf0859ef8c523cad68`.
"""
    Path("CHANGELOG.md").write_text(changelog, encoding="utf-8")
    write_json(
        target / "publication_provenance.json",
        {
            "created_at_utc": utc_now(),
            "v1_pooled": False,
            "reporting_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "report_source_sha256": file_hash(__file__),
            "namespace": config.namespace,
            "main_models": verification["main_models"],
            "english_chinese_same_numeric_table": True,
        },
    )
    print("Wrote actual-data report, bilingual READMEs, changelog and offline dataset.")


def export_demo(config, target, verification):
    summary = pd.read_csv(target / "main_summary.csv")
    variants = pd.read_csv(target / "variants.csv")
    curves = pd.read_csv(target / "learning_curves.csv")
    main_rows = [
        {
            "distance": int(row.distance),
            "n": int(row.n),
            "blocked": row.blocked_pct,
            "interleaved": row.interleaved_pct,
            "difference": row.difference_pp,
            "ci_low": row.ci_low_pp,
            "ci_high": row.ci_high_pp,
        }
        for row in summary.itertuples()
    ]
    variants["mean_accuracy"] *= 100
    selected = curves[
        (curves.stage == "main")
        & (curves.split == "noisy")
        & (curves.group == "all")
        & curves.schedule.isin(["blocked", "interleaved"])
    ].copy()
    selected[["mean", "se"]] *= 100
    example_rows = []
    for job in make_jobs(config, "main"):
        if job["run_id"] == 0:
            manifest = json.loads((cell_directory(config, job) / "manifest.json").read_text())
            example_rows.append(
                {
                    "distance": job["distance"],
                    "bases": [
                        manifest["main_sequences"][letter]["actions_0_based"] for letter in "ABC"
                    ],
                    "changed": manifest["changed_positions"],
                }
            )
    payload = {
        "provenance": {
            "Study": "v2 planned scientific suite",
            "Namespace": config.namespace,
            "Scientific main models": verification["main_models"],
            "Primary pairs per distance": config.repetitions,
            "Exploratory pairs per original distance": config.exploratory_repetitions,
            "Producing Git": ", ".join(verification["producing_git_commits"]),
            "Unit": "independent replicate ID",
            "Prior v1 data pooled": "No",
        },
        "main": main_rows,
        "variants": variants.to_dict("records"),
        "curves": selected[
            ["architecture", "hidden_size", "distance", "schedule", "update", "mean", "se"]
        ].to_dict("records"),
        "examples": example_rows,
    }
    Path("docs/demo/data.js").write_text(
        "// Generated only from verified, saved v2 scientific results.\nwindow.CRUX_DATA = "
        + json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
