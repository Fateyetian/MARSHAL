"""
State-conditional advantage (AWM 阶段一).

核心思想：把 MARSHAL 的"按角色全局归一化"换成"按状态局部归一化 + 跨 rollout V-table 基线"，
即在 turn 层面计算 advantage: A_k = R_k - b(s_k)，
其中 b(s_k) = alpha * batch内同状态均值 + (1-alpha) * V(s_k)（hybrid）。

本模块提供：
1. StateValueTable：跨 step 的状态价值表（内存中），用 EMA 更新。
2. tictactoe_coarse_state_id：从 OpenSpiel observation_tensor 提取粗粒度状态哈希。

集成点见 roll/pipeline/agentic/agentic_pipeline.py 与 roll/agentic/rollout/env_manager.py。
默认关闭（config: state_value.enabled=False），不影响 baseline。
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

# tic-tac-toe 的 8 条获胜线
_TTT_WIN_LINES: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # 行
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # 列
    (0, 4, 8), (2, 4, 6),  # 对角
)


def tictactoe_coarse_state_id(observation_tensor: Sequence[float], current_player: int) -> int:
    """从 tic-tac-toe 的 observation_tensor 提取粗粒度状态哈希。

    OpenSpiel tic_tac_toe 的 observation_tensor 长度 27 = 3 通道 × 9 格，通道固定（非视角相对）：
        ch0(0-8)   : 空格掩码
        ch1(9-17)  : player1 的棋子
        ch2(18-26) : player0 的棋子

    特征（相对当前行动方 current_player，对称、状态空间减半利于 V-table 覆盖）：
        num_pieces : 总落子数（0-9，游戏阶段）
        center     : 中心格(cell4)归属 {0:空, 1:我, 2:对手}
        my_threats : 我方"二缺一"威胁线数
        opp_threats: 对手"二缺一"威胁线数

    Returns:
        稳定的整数 state_id。
    """
    obs = np.asarray(observation_tensor, dtype=np.float32).reshape(3, 9)
    empty = obs[0]
    p1 = obs[1]
    p0 = obs[2]
    if current_player == 0:
        mine, opp = p0, p1
    else:
        mine, opp = p1, p0

    # 游戏阶段
    num_pieces = int(round((1.0 - empty).sum()))

    # 中心格归属
    if empty[4] > 0.5:
        center = 0
    elif mine[4] > 0.5:
        center = 1
    else:
        center = 2

    # 威胁线数（某方恰有 2 子且第三格空）
    my_threats = 0
    opp_threats = 0
    for a, b, c in _TTT_WIN_LINES:
        line_mine = mine[a] + mine[b] + mine[c]
        line_opp = opp[a] + opp[b] + opp[c]
        line_empty = empty[a] + empty[b] + empty[c]
        if line_mine >= 1.5 and line_empty >= 0.5:  # 我方2子 + 1空
            my_threats += 1
        if line_opp >= 1.5 and line_empty >= 0.5:  # 对手2子 + 1空
            opp_threats += 1

    return hash((num_pieces, center, my_threats, opp_threats))


class StateValueTable:
    """跨 rollout 的状态价值表（内存中、不持久化）。

    用 EMA 更新：V_new(s) = (1 - eta) * V_old(s) + eta * mean_R_current_batch(s)。
    提供 hybrid baseline：b(s) = alpha * batch_mean(s) + (1-alpha) * V(s)，
    当某状态在 batch 内出现次数 < min_count 时，回退到全局均值（避免单样本方差爆炸）。
    """

    def __init__(self, ema_eta: float = 0.1, min_count: int = 2, alpha: float = 0.5):
        self.ema_eta = ema_eta
        self.min_count = min_count
        self.alpha = alpha
        self.values: Dict[int, float] = {}
        self.global_mean: float = 0.0
        self.global_count: int = 0

    def reset(self) -> None:
        self.values.clear()
        self.global_mean = 0.0
        self.global_count = 0

    def _batch_mean_by_state(self, state_ids: List[int], rewards: np.ndarray) -> Tuple[Dict[int, float], Dict[int, int]]:
        """按 state_id 聚合本 batch 的平均回报与计数。"""
        sums: Dict[int, float] = {}
        counts: Dict[int, int] = {}
        for sid, r in zip(state_ids, rewards):
            sums[sid] = sums.get(sid, 0.0) + float(r)
            counts[sid] = counts.get(sid, 0) + 1
        batch_mean = {sid: sums[sid] / counts[sid] for sid in sums}
        return batch_mean, counts

    def compute_baselines(self, state_ids: List[int], rewards: np.ndarray) -> np.ndarray:
        """为每个 turn 计算 hybrid baseline b(s_k)。

        b(s) = alpha * batch_mean(s) + (1-alpha) * V(s)
        当 batch 内 s 出现次数 < min_count，batch_mean 项回退到全局均值；
        当 V(s) 未见过，V 项回退到全局均值。
        """
        rewards = np.asarray(rewards, dtype=np.float32)
        if len(state_ids) == 0:
            return np.zeros_like(rewards)
        batch_mean, counts = self._batch_mean_by_state(state_ids, rewards)

        # 更新全局均值（用本 batch 的所有 turn 回报）
        self.global_count += len(rewards)
        if self.global_count > 0:
            batch_global = float(rewards.mean())
            # 全局均值也用 EMA，平滑跨 step
            self.global_mean = (1 - self.ema_eta) * self.global_mean + self.ema_eta * batch_global

        baselines = np.empty(len(state_ids), dtype=np.float32)
        for i, sid in enumerate(state_ids):
            bm = batch_mean[sid] if counts.get(sid, 0) >= self.min_count else self.global_mean
            v = self.values.get(sid, self.global_mean)
            baselines[i] = self.alpha * bm + (1.0 - self.alpha) * v
        return baselines

    def update(self, state_ids: List[int], rewards: np.ndarray) -> None:
        """step 末调用：用本 batch 各状态的平均回报 EMA 更新 V-table。"""
        rewards = np.asarray(rewards, dtype=np.float32)
        if len(state_ids) == 0:
            return
        batch_mean, _ = self._batch_mean_by_state(state_ids, rewards)
        for sid, mean_r in batch_mean.items():
            old = self.values.get(sid, self.global_mean)
            self.values[sid] = (1.0 - self.ema_eta) * old + self.ema_eta * mean_r

    def __len__(self) -> int:
        return len(self.values)
