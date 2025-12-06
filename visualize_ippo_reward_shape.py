"""可视化 IPPO 个体奖励随车道距离 d 变化的曲线

运行方法：
    cd d:\lxpaper\DRL\ippo
    python visualize_ippo_reward_shape.py

图中 d 表示当前车道与目标车道的距离（以“车道数”为单位），
当前 IPPO 奖励实现为：
    r(d) = -α d^2 - β d + exp(-d^2/σ^2) - 0.5 - time_penalty
参数取自 ippo_trainer.IPPOTrainer._compute_individual_rewards。
"""

import numpy as np
import matplotlib.pyplot as plt


def ippo_reward_function(d: np.ndarray,
                         alpha: float = 0.3,
                         beta: float = 0.1,
                         sigma: float = 0.5,
                         time_penalty: float = 0.02) -> dict:
    """根据当前实现计算奖励各部分和总和。

    参数
    -----
    d : np.ndarray
        当前车道与目标车道的距离（单位：车道数，连续值）。
    alpha, beta, sigma, time_penalty : float
        与 ippo_trainer 中保持一致的超参数。
    """
    # 势能项 φ(d) = -α d^2 - β d
    potential = -alpha * d ** 2 - beta * d

    # 高斯项 exp(-d^2 / σ^2) - 0.5
    gaussian = np.exp(-(d ** 2) / (sigma ** 2)) - 0.5

    # 时间惩罚
    time_cost = -time_penalty * np.ones_like(d)

    total = potential + gaussian + time_cost

    return {
        "potential": potential,
        "gaussian": gaussian,
        "time_cost": time_cost,
        "total": total,
    }


def main() -> None:
    # 车道距离 d：从 0 到 3 车道，包含中间的连续值
    d = np.linspace(0.0, 3.0, 301)

    parts = ippo_reward_function(d)

    # 字体设置：尽量兼容中文
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(d, parts["total"], label="总奖励 r(d)", color="#1f77b4", linewidth=2)
    ax.plot(d, parts["potential"], "--", label="势能项 -αd²-βd", color="#ff7f0e")
    ax.plot(d, parts["gaussian"], "-.", label="高斯项 exp(-d²/σ²)-0.5", color="#2ca02c")
    ax.plot(d, parts["time_cost"], ":", label="时间惩罚", color="#d62728")

    # 标出整数车道距离的位置
    for k in range(0, 4):
        y = ippo_reward_function(np.array([k]))["total"][0]
        ax.scatter([k], [y], color="black")
        ax.text(k, y, f" d={k}", fontsize=9, ha="left", va="bottom")

    ax.set_xlabel("车道距离 d (|current_lane - target_lane|)")
    ax.set_ylabel("单步奖励 r(d)")
    ax.set_title("IPPO 个体奖励随车道距离的变化")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
