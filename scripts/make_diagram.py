#!/usr/bin/env python
"""System overview diagram -> docs/figures/system.png.

    uv run --with matplotlib python scripts/make_diagram.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams["font.family"] = ["Hiragino Sans", "sans-serif"]
VLM, POL, DATA, IO = "#2a9d8f", "#264653", "#e9c46a", "#8d99ae"


def box(ax, x, y, w, h, title, sub, col, fg="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=col, ec="none"))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=12, color=fg, weight="bold")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=9, color=fg)


def arrow(ax, a, b, text="", rad=0.0, col="#444", dy=0.18, dx=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.8, color=col,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, text, ha=ha, va="bottom", fontsize=9, color=col)


def main() -> None:
    fig, ax = plt.subplots(figsize=(12.8, 6.2), dpi=150)
    ax.set_xlim(-0.1, 16.2); ax.set_ylim(0, 8.1); ax.axis("off")

    # top row: one instruction -> plan -> point -> act -> judge
    box(ax, 0.0, 5.2, 2.7, 1.6, "指示", "「片付けて」\n言い換え・否定・英語も可", IO)
    box(ax, 3.4, 5.2, 2.8, 1.6, "計画役 (Claude)", "正面+横の画像から\n「どれをどの箱へ」の手順列", VLM)
    box(ax, 7.0, 5.2, 2.6, 1.6, "指し示し (Claude)", "画像上で 物体 と\n置き場所 を点で示す", VLM)
    box(ax, 10.3, 5.2, 3.0, 1.6, "ACT 方策", "カメラ 2 台 + 指された 2 点\n→ 腕の動作 (10 Hz)", POL)
    box(ax, 13.5, 5.2, 2.5, 1.6, "判定役 (Claude)", "正面カメラ 6 コマで\n成否を判定", VLM)
    arrow(ax, (2.7, 6.0), (3.4, 6.0))
    arrow(ax, (6.2, 6.0), (7.0, 6.0))
    arrow(ax, (9.6, 6.0), (10.3, 6.0))
    arrow(ax, (13.3, 6.0), (13.5, 6.0))
    arrow(ax, (14.7, 6.8), (4.8, 6.8), "成功 → 次の手順 / 全部終わったら画像を見て再計画", rad=0.12, col=VLM, dy=0.72)

    # bottom row: failure -> teacher -> data -> audit -> retrain
    box(ax, 13.2, 2.5, 2.8, 1.6, "教師が引き継ぎ", "失敗した配置から続きを実演\n(実機なら人の遠隔操作)", "#e76f51")
    box(ax, 9.6, 2.5, 3.0, 1.6, "デモとして保存", "倒れた円柱・壁際・積み重なり\nなど苦手な状態の立て直し", DATA, fg="#222")
    box(ax, 5.6, 2.5, 3.2, 1.6, "データ監査 (Claude)", "指示と動作が食い違うデモを\n学習前に除く(別実験で評価)", VLM)
    box(ax, 1.4, 2.5, 3.4, 1.6, "再学習", "次の版の方策へ\n(失敗の発見はカメラと VLM だけ)", POL)
    arrow(ax, (14.75, 5.2), (14.6, 4.1), "失敗", col="#e76f51", dy=-0.15, dx=0.2, ha="left")
    arrow(ax, (13.2, 3.3), (12.6, 3.3))
    arrow(ax, (9.6, 3.3), (8.8, 3.3))
    arrow(ax, (5.6, 3.3), (4.8, 3.3))
    arrow(ax, (3.1, 4.1), (11.7, 5.2), "方策を更新", col=POL, dy=0.12)
    ax.text(8.0, 0.9, "配備しながら、失敗を自分で見つけて学び直す ― データフライホイール", ha="center", fontsize=14,
            color="#222", weight="bold")
    ax.text(8.0, 0.35, "シミュレータ: MuJoCo / Franka Panda    実装: ROS 2 Jazzy のノード (計画・指し示し・方策・判定・可視化)",
            ha="center", fontsize=10, color="#555")
    out = Path("docs/figures/system.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
