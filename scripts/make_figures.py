#!/usr/bin/env python
"""Report figures from the evaluation JSONs in outputs/eval -> docs/figures/*.png.

    uv run --with matplotlib python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "sans-serif"]
EVAL, OUT = Path("outputs/eval"), Path("docs/figures")
MODES = [("success", "成功", "#2a9d8f"), ("no_grasp", "掴めず", "#e76f51"), ("contact", "衝突", "#f4a261"),
         ("dropped", "途中で落下", "#e9c46a"), ("missed_bin", "箱を外す", "#8d99ae"),
         ("timeout_holding", "時間切れ", "#adb5bd"), ("wrong_object", "違う物体", "#6d597a")]


def load(tag: str) -> dict | None:
    p = EVAL / f"{tag}.json"
    return json.loads(p.read_text()) if p.exists() else None


def flywheel() -> None:
    arms = [("VLM の判定だけ(内部情報なし)", ["b10-v0", "b10-vj1", "b10-vj2", "b10-vj3"], "#2a9d8f", "o"),
            ("シミュレータの正解", ["b10-v0", "b10-v1", "b10-v2", "b10-v3"], "#264653", "s"),
            ("対照: 通常デモを同数追加", ["b10-v0", "b10-v1c", "b10-v2c", "b10-v3c"], "#adb5bd", "^")]
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for k, (name, tags, col, mk) in enumerate(arms):
        xs, ys, lo, hi = [], [], [], []
        for i, t in enumerate(tags):
            d = load(t)
            if d is None:
                continue
            xs.append(i + (k - 1) * 0.06); ys.append(100 * d["success_rate"])
            lo.append(100 * (d["success_rate"] - d["wilson95"][0])); hi.append(100 * (d["wilson95"][1] - d["success_rate"]))
        ax.errorbar(xs, ys, yerr=[lo, hi], color=col, marker=mk, capsize=3, lw=2 if k == 0 else 1.3, label=name)
    ax.set_xticks(range(4), ["v0\n(教師 10 本)", "v1", "v2", "v3"])
    ax.set_ylabel("成功率 [%](未使用の 100 配置、95% 区間)")
    ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
    ax.set_title("データフライホイール: 反復ごとの成功率")
    fig.tight_layout(); fig.savefig(OUT / "flywheel.png"); plt.close(fig)


def modes(tags: list[tuple[str, str]], name: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    for i, (t, label) in enumerate(tags):
        d = load(t)
        if d is None:
            continue
        left = 0
        for key, jp, col in MODES:
            v = d["failure_modes"].get(key, 0)
            ax.barh(i, v, left=left, color=col, label=jp if i == 0 else None)
            if v >= 6:
                ax.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=8, color="white")
            left += v
    ax.set_yticks(range(len(tags)), [l for _, l in tags]); ax.invert_yaxis()
    ax.set_xlabel("回数(100 回中)"); ax.set_xlim(0, 100); ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.25), frameon=False)
    fig.tight_layout(); fig.savefig(OUT / name); plt.close(fig)


def deploy() -> None:
    """Held-out tidy-all over the rounds of the deployment flywheel: how many of the 3 objects ended up in a box."""
    rows = []
    for k in range(10):
        ps = [Path("outputs/tidy_all") / f"d200-v{k}{x}.json" for x in ("", "_ext_a", "_ext_b")]  # 1000-1039, 1040-1099
        if all(p.exists() for p in ps):
            rows.append((f"v{k}", [e["tidied"] for p in ps for e in json.loads(p.read_text())["episodes"]]))
    cols = {3: "#2a9d8f", 2: "#e9c46a", 1: "#f4a261", 0: "#e76f51"}
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
    for i, (name, ts) in enumerate(rows):
        left = 0
        for c in (3, 2, 1, 0):
            v = 100 * sum(t == c for t in ts) / len(ts)
            ax.barh(i, v, left=left, color=cols[c], label=f"{c} 個片付いた" if i == 0 else None)
            if v >= 6:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white")
            left += v
    ax.set_yticks(range(len(rows)), [f"{n}\n(介入を {k} 周学習)" if k else f"{n}\n(片付け途中デモ 200 本)"
                                     for k, (n, _) in enumerate(rows)])
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("場面の割合 [%](未使用の 100 場面)")
    ax.set_title("「片付けて」配備中の介入で学び直した結果")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.3), frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "deploy_flywheel.png"); plt.close(fig)


def pair(seed: int = 1000, n: int = 5) -> None:
    """Same held-out placement: v0 fails, v3 (VLM-only flywheel) succeeds. Frames sampled over each episode."""
    d0, d3 = load("b10-v0"), load("b10-vj3")
    if d0 is None or d3 is None:
        return
    rows = []
    for d, name in ((d0, "v0"), (d3, "v3")):
        e = next(e for e in d["episodes"] if e["seed"] == seed)
        mode = next(jp for key, jp, _ in MODES if key == e["failure_mode"])
        rows.append((f"{name}: {mode}", imageio.mimread(e["video"], memtest=False)))
    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 3.6), dpi=150)
    for r, (title, frames) in enumerate(rows):
        idx = [round(i * (len(frames) - 1) / (n - 1)) for i in range(n)]
        for c, i in enumerate(idx):
            ax = axes[r, c]; ax.imshow(frames[i]); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{i / 10:.0f} 秒", fontsize=8, pad=2)
        axes[r, 0].set_ylabel(title, fontsize=10, weight="bold", color="#e76f51" if r == 0 else "#2a9d8f")
    fig.suptitle(f"同じ配置(評価場面 {seed})での v0(教師 10 本)と v3(VLM の判定だけで 3 周)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "flywheel_pair.png"); plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    flywheel()
    modes([("b10-v0", "v0"), ("b10-vj1", "v1 (VLM)"), ("b10-vj2", "v2 (VLM)"), ("b10-vj3", "v3 (VLM)")],
          "flywheel_modes.png", "VLM 系列の結果の内訳")
    modes([("m3-v0", "正しいデモ 100 本"), ("m3-p30", "30% すり替え"), ("m3-p30a", "監査で除去(70 本)")],
          "audit.png", "指示と動作が食い違うデモの影響と、VLM 監査による回復")
    deploy()
    pair()
    print(sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
