# -*- coding: utf-8 -*-
"""
非 Transformer 阴阳双态 DAO 认知核心 (DaoCognitiveCore)
======================================================

核心算法与动力学方程 (对应大兮 AGI V14-V20 架构规范):
1. 局部稀疏原型路由 + 良心稳态 (Conscience Homeostasis, A20):
   - 输入投影: z_t = W_x x_t  (或直接接收归一化概念/联想特征向量)
   - 带良心偏置的原型距离: d_i = ||z_t - p_i||_2^2 + theta_i
   - 稀疏活跃支撑集: I_t = TopK(-d, K)  (复杂度 O(N_p * D_z), 无 O(L^2) 自注意力)
   - 局部归一化路由权重: q_{t,i} = softmax(-d_{I_t} / tau)
   - 使用率指数滑动平均: u_i <- (1 - lambda_u) u_i + lambda_u * 1[i in I_t]
   - 良心稳态偏置更新: theta_i <- clip(theta_i + eta_h * (u_i - u^*), -theta_max, theta_max), u^* = K / N_hot

2. 阴阳双态循环动力学 (Dual-Path Yin-Yang Dynamics):
   - 阴态激活 (兴奋性记忆整合流):
     y_{t+1} = sigma(alpha_y * y_t + beta_y * q_t - gamma_{yn} * n_t + b_y)
   - 阳态抑制 (预测误差驱动的制动门控流):
     n_{t+1} = sigma(alpha_n * n_t + beta_n * r_t + gamma_{ny} * y_t + b_n)
     其中零误差基线满足 b_n = -12.0, gamma_{ny} = 0.0, beta_n > 0 => 当 r_t = 0 时 n_{t+1} < 1e-5 (零误伤率)
   - 原型记忆读出:
     m_t = sum_{i in I_t} q_{t,i} * y_{t+1,i} * p_i
   - 状态循环更新:
     s_{t+1} = tanh(A * s_t + B * z_t + G * m_t)
   - 阳态门控输出:
     y_hat_t = W_o * (s_{t+1} * (1 - kappa * mean(n_{t+1, I_t})))

3. 三因子局部可塑性与双系统互补睡眠巩固 (Three-Factor Local Plasticity & Sleep Consolidation):
   - 资格迹 (Eligibility Trace):
     e_{t,i} = lambda_e * e_{t-1,i} + q_{t,i} * y_{t+1,i} * (z_t - p_i),  forall i in I_t
   - 神经调质信号 (Dopaminergic / Surprise Modulator):
     M_t = clip(r_scalar - r_bar, -M_max, M_max)
   - 局部原型可塑性更新 (仅在稀疏活跃集 I_t 上更新):
     Delta p_i = eta_p * M_t * e_{t,i}
   - 突触稳态缩放 (Synaptic Scaling):
     ||p_i||_2 <= R_max
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import time
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


@dataclass
class DaoConfig:
    n_prototypes: int = 128          # 原型槽位数 N_p
    n_hot: int = 128                 # 热原型目标数 N_hot
    dim_z: int = 64                  # 原型与隐状态维度 D_z
    top_k: int = 8                   # 稀疏激活数 K (K << N_p)
    tau: float = 0.25                # 路由温度系数
    # 良心稳态参数 (A20)
    lambda_u: float = 0.05           # 使用率 EMA 衰减率
    eta_h: float = 0.20              # 良心偏置步长
    theta_max: float = 2.5           # 良心偏置饱和上限
    # 阴阳双态动力学参数
    alpha_y: float = 0.55            # 阴态自保持系数
    beta_y: float = 3.20             # 阴态路由驱动增益
    gamma_yn: float = 2.40           # 阳对阴的交叉抑制增益
    b_y: float = -0.35               # 阴态静息偏置
    alpha_n: float = 0.30            # 阳态自保持系数
    beta_n: float = 14.0             # 阳态预测误差敏感增益
    gamma_ny: float = 0.0            # 阴对阳耦合 (设为0保证零误差时阳态严格归零)
    b_n: float = -12.0               # 阳态静息偏置 (保证 r_t=0 时 sigma(b_n) < 1e-5)
    kappa: float = 0.85              # 阳态输出门控抑制强度
    # 三因子局部可塑性与稳态缩放
    lambda_e: float = 0.75           # 资格迹衰减因子
    eta_p: float = 0.08              # 局部可塑性学习率
    m_max: float = 1.5               # 调制信号裁剪阈值
    r_max: float = 2.0               # 原型范数稳态上限
    episodic_capacity: int = 512     # 海马体快速情景缓冲容量
    seed: int = 42


class DaoCognitiveCore:
    """非 Transformer 阴阳双态认知核心"""

    def __init__(self, cfg: Optional[DaoConfig] = None):
        self.cfg = cfg or DaoConfig()
        rng = np.random.RandomState(self.cfg.seed)
        np_cnt = self.cfg.n_prototypes
        dz = self.cfg.dim_z

        # 原型记忆库 P: (N_p, D_z), 单位球面初始化
        raw_p = rng.normal(0.0, 1.0, size=(np_cnt, dz)).astype(np.float32)
        norms = np.linalg.norm(raw_p, axis=1, keepdims=True) + 1e-8
        self.prototypes: np.ndarray = (raw_p / norms).astype(np.float32)
        self.prototype_labels: List[str] = [f"proto_{i}" for i in range(np_cnt)]

        # 良心稳态状态: 使用率 EMA u_i 与偏置 theta_i
        u_star = float(self.cfg.top_k) / float(max(1, self.cfg.n_hot))
        self.usage_ema: np.ndarray = np.full(np_cnt, u_star, dtype=np.float32)
        self.theta: np.ndarray = np.zeros(np_cnt, dtype=np.float32)

        # 阴阳双态向量与循环隐状态
        self.yin_state: np.ndarray = np.zeros(np_cnt, dtype=np.float32)
        self.yang_state: np.ndarray = np.zeros(np_cnt, dtype=np.float32)
        self.recurrent_state: np.ndarray = np.zeros(dz, dtype=np.float32)

        # 资格迹 E: (N_p, D_z) 与误差基线 r_bar
        self.eligibility: np.ndarray = np.zeros((np_cnt, dz), dtype=np.float32)
        self.reward_baseline: float = 0.15

        # 循环与输出权重 (对角+低秩稳定初始化, 避免密集大矩阵)
        self.diag_A: np.ndarray = np.full(dz, 0.45, dtype=np.float32)
        self.diag_B: np.ndarray = np.full(dz, 0.55, dtype=np.float32)
        self.diag_G: np.ndarray = np.full(dz, 0.65, dtype=np.float32)

        # 海马体快速情景缓存 (用于睡眠重放巩固)
        self.episodic_buffer: List[Dict[str, Any]] = []
        self.step_count: int = 0
        self.sleep_cycles: int = 0
        self.split_merge_events: int = 0
        self.last_telemetry: Dict[str, Any] = {}

    def project_vector(self, vec: np.ndarray) -> np.ndarray:
        """将任意维度的输入特征向量确定性哈希折叠投影到 D_z 维空间并归一化。"""
        v = np.asarray(vec, dtype=np.float32).ravel()
        dz = self.cfg.dim_z
        if v.size == 0:
            return np.zeros(dz, dtype=np.float32)
        if v.size == dz:
            z = v.copy()
        elif v.size > dz:
            # 分块折叠求和 (确定性保距投影)
            pad = (-v.size) % dz
            if pad:
                v = np.pad(v, (0, pad), mode="constant")
            z = v.reshape(-1, dz).sum(axis=0)
        else:
            z = np.zeros(dz, dtype=np.float32)
            z[:v.size] = v
        norm = float(np.linalg.norm(z))
        if norm > 1e-8:
            z = z / norm
        return z.astype(np.float32)

    def route_sparse(
        self, z: np.ndarray, update_homeostasis: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """局部稀疏原型路由 + 良心稳态 (Conscience Homeostasis, A20)。
        返回: (active_indices I_t, active_weights q_{t,I_t}, full_weights q_t)
        """
        # d_i = ||z - p_i||_2^2 + theta_i
        diffs = self.prototypes - z[None, :]
        sq_dists = np.einsum("ij,ij->i", diffs, diffs)
        eff_dists = sq_dists + self.theta

        k = min(self.cfg.top_k, self.cfg.n_prototypes)
        if k >= self.cfg.n_prototypes:
            top_idx = np.argsort(eff_dists)
        else:
            part = np.argpartition(eff_dists, k)[:k]
            top_idx = part[np.argsort(eff_dists[part])]

        # Softmax over -d_{I_t} / tau
        logits = -eff_dists[top_idx] / max(1e-4, self.cfg.tau)
        logits = logits - np.max(logits)
        exp_l = np.exp(logits)
        q_active = (exp_l / (np.sum(exp_l) + 1e-12)).astype(np.float32)

        q_full = np.zeros(self.cfg.n_prototypes, dtype=np.float32)
        q_full[top_idx] = q_active

        if update_homeostasis:
            hit_mask = np.zeros(self.cfg.n_prototypes, dtype=np.float32)
            hit_mask[top_idx] = 1.0
            lu = self.cfg.lambda_u
            self.usage_ema = (1.0 - lu) * self.usage_ema + lu * hit_mask
            u_star = float(k) / float(max(1, self.cfg.n_hot))
            self.theta = np.clip(
                self.theta + self.cfg.eta_h * (self.usage_ema - u_star),
                -self.cfg.theta_max,
                self.cfg.theta_max,
            ).astype(np.float32)

        return top_idx, q_active, q_full

    def step(
        self,
        input_vec: np.ndarray,
        prediction_error: float = 0.0,
        learn: bool = True,
        topic_label: Optional[str] = None,
    ) -> Dict[str, Any]:
        """执行一步完整的 DAO 阴阳双态循环与三因子局部可塑性更新。"""
        z = self.project_vector(input_vec)
        top_idx, q_active, q_full = self.route_sparse(z, update_homeostasis=True)

        # 预测误差向量 r_t (在活跃集上分布预测误差, 零误差时全为 0)
        r_scalar = float(max(0.0, prediction_error))
        r_vec = np.zeros(self.cfg.n_prototypes, dtype=np.float32)
        if r_scalar > 1e-9:
            r_vec[top_idx] = r_scalar * q_active

        # 1. 阳态更新 (先算阳态以体现零误差精确归零, 有误差立即制动)
        if r_scalar <= 1e-9 and float(np.max(self.yang_state)) < 1e-4:
            # 严格零误差基线: b_n = -12 => sigma(-12) = 6.1e-6
            yang_next = _sigmoid(np.full(self.cfg.n_prototypes, self.cfg.b_n, dtype=np.float32))
        else:
            yang_drive = (
                self.cfg.alpha_n * self.yang_state
                + self.cfg.beta_n * r_vec
                + self.cfg.gamma_ny * self.yin_state
                + self.cfg.b_n
            )
            yang_next = _sigmoid(yang_drive).astype(np.float32)

        # 2. 阴态更新 (受路由权重 q_full 激励, 受阳态 yang_next 抑制)
        yin_drive = (
            self.cfg.alpha_y * self.yin_state
            + self.cfg.beta_y * q_full
            - self.cfg.gamma_yn * yang_next
            + self.cfg.b_y
        )
        yin_next = _sigmoid(yin_drive).astype(np.float32)

        self.yin_state = yin_next
        self.yang_state = yang_next

        # 3. 稀疏原型记忆读出 m_t = sum_{i in I_t} q_{t,i} * y_{t+1,i} * p_i
        active_weights = q_active * yin_next[top_idx]
        m_t = np.sum(self.prototypes[top_idx] * active_weights[:, None], axis=0).astype(np.float32)

        # 4. 循环隐状态更新 s_{t+1} = tanh(A s_t + B z_t + G m_t)
        s_next = np.tanh(
            self.diag_A * self.recurrent_state
            + self.diag_B * z
            + self.diag_G * m_t
        ).astype(np.float32)
        self.recurrent_state = s_next

        # 5. 阳态门控输出 y_hat_t = s_{t+1} * (1 - kappa * mean(n_{t+1, I_t}))
        yang_active_mean = float(np.mean(yang_next[top_idx]))
        gate = float(np.clip(1.0 - self.cfg.kappa * yang_active_mean, 0.05, 1.0))
        y_hat = (s_next * gate).astype(np.float32)

        # 6. 三因子局部可塑性 (仅更新活跃原型集 I_t)
        modulator = 0.0
        if learn:
            # 资格迹衰减 + 局部 Hebbian 项 q_{t,i} * y_{t+1,i} * (z_t - p_i)
            self.eligibility *= self.cfg.lambda_e
            for pos, proto_idx in enumerate(top_idx):
                hebb_drive = float(q_active[pos] * yin_next[proto_idx])
                self.eligibility[proto_idx] += hebb_drive * (z - self.prototypes[proto_idx])
                if topic_label and hebb_drive > 0.15:
                    self.prototype_labels[proto_idx] = topic_label[:48]

            # 调制信号 M_t: 低误差确认或适度新颖性驱动原型向输入吸引
            # 当输入是有价值的新经验时, 正向聚合到最匹配原型
            novelty_signal = float(1.0 - min(1.0, r_scalar))
            modulator = float(np.clip(
                0.35 + 0.65 * (novelty_signal - self.reward_baseline),
                -self.cfg.m_max,
                self.cfg.m_max,
            ))
            self.reward_baseline = 0.95 * self.reward_baseline + 0.05 * novelty_signal

            # 仅对 I_t 执行局部更新 + 突触稳态缩放
            for proto_idx in top_idx:
                self.prototypes[proto_idx] += self.cfg.eta_p * modulator * self.eligibility[proto_idx]
                p_norm = float(np.linalg.norm(self.prototypes[proto_idx]))
                if p_norm > self.cfg.r_max:
                    self.prototypes[proto_idx] *= (self.cfg.r_max / p_norm)
                elif p_norm > 1e-8:
                    # 保持在单位球面附近以稳定余弦/欧氏路由
                    self.prototypes[proto_idx] /= p_norm

            # 写入海马体快速情景缓存
            self.episodic_buffer.append({
                "z": z,
                "error": r_scalar,
                "topic": topic_label or "",
                "ts": time.time(),
            })
            if len(self.episodic_buffer) > self.cfg.episodic_capacity:
                self.episodic_buffer.pop(0)

        self.step_count += 1
        lyapunov_v = float(r_scalar ** 2 + 0.5 * float(np.mean(yang_next[top_idx] ** 2)))
        telemetry = {
            "step": self.step_count,
            "active_prototypes": [int(i) for i in top_idx],
            "active_labels": [self.prototype_labels[int(i)] for i in top_idx[:4]],
            "top_weight": round(float(q_active[0]), 4) if len(q_active) else 0.0,
            "yin_mean": round(float(np.mean(yin_next[top_idx])), 4),
            "yang_mean": round(yang_active_mean, 6),
            "yang_gate": round(gate, 4),
            "prediction_error": round(r_scalar, 4),
            "modulator_M": round(modulator, 4),
            "usage_entropy": round(self.usage_entropy(), 4),
            "lyapunov_energy": round(lyapunov_v, 6),
            "output_norm": round(float(np.linalg.norm(y_hat)), 4),
        }
        self.last_telemetry = telemetry
        return {
            "z": z,
            "y_hat": y_hat,
            "active_indices": top_idx,
            "active_weights": q_active,
            "telemetry": telemetry,
        }

    def usage_entropy(self) -> float:
        """计算原型使用率的归一化香农熵 H(u) in [0, 1], 衡量良心稳态防坍缩效果。"""
        u = np.maximum(self.usage_ema, 1e-12)
        p = u / np.sum(u)
        h = -float(np.sum(p * np.log(p)))
        max_h = math.log(max(2, self.cfg.n_prototypes))
        return float(np.clip(h / max_h, 0.0, 1.0))

    def sleep_consolidate(self, n_replay: int = 32, lr: float = 0.03, replay_batches: Optional[int] = None) -> Dict[str, Any]:
        """双系统互补睡眠巩固: 从海马体情景缓存重放高惊奇/高价值样本并执行结构分裂/合并。"""
        if replay_batches is not None:
            n_replay = int(replay_batches)
        if not self.episodic_buffer:
            ent = round(self.usage_entropy(), 4)
            return {"replayed": 0, "split_merged": 0, "usage_entropy": ent, "usage_entropy_ratio": ent}

        # 按惊奇度/预测误差加权采样重放
        episodes = list(self.episodic_buffer)
        episodes.sort(key=lambda x: x.get("error", 0.0), reverse=True)
        selected = episodes[:min(n_replay, len(episodes))]

        for ep in selected:
            z = ep["z"]
            top_idx, q_active, _ = self.route_sparse(z, update_homeostasis=True)
            for pos, idx in enumerate(top_idx[:3]):
                w = float(q_active[pos]) * lr
                self.prototypes[idx] = (1.0 - w) * self.prototypes[idx] + w * z
                nrm = float(np.linalg.norm(self.prototypes[idx]))
                if nrm > 1e-8:
                    self.prototypes[idx] /= nrm

        # 结构可塑性: 若最热原型使用率超过 4 * u^* 且最冷原型接近死区, 执行分裂-再生
        u_star = float(self.cfg.top_k) / float(max(1, self.cfg.n_hot))
        max_i = int(np.argmax(self.usage_ema))
        min_i = int(np.argmin(self.usage_ema))
        split_merged = 0
        if self.usage_ema[max_i] > 3.5 * u_star and self.usage_ema[min_i] < 0.15 * u_star:
            noise = np.random.RandomState(self.step_count + 17).normal(
                0.0, 0.05, size=self.cfg.dim_z
            ).astype(np.float32)
            new_p = self.prototypes[max_i] + noise
            new_p /= (np.linalg.norm(new_p) + 1e-8)
            self.prototypes[min_i] = new_p
            self.prototype_labels[min_i] = f"{self.prototype_labels[max_i]}#split"
            half_u = 0.5 * (self.usage_ema[max_i] + self.usage_ema[min_i])
            self.usage_ema[max_i] = half_u
            self.usage_ema[min_i] = half_u
            self.theta[max_i] = 0.0
            self.theta[min_i] = 0.0
            self.split_merge_events += 1
            split_merged = 1

        self.sleep_cycles += 1
        ent = round(self.usage_entropy(), 4)
        return {
            "replayed": len(selected),
            "sleep_cycles": self.sleep_cycles,
            "split_merged": split_merged,
            "usage_entropy": ent,
            "usage_entropy_ratio": ent,
        }

    def stats(self) -> Dict[str, Any]:
        u_star = float(self.cfg.top_k) / float(max(1, self.cfg.n_hot))
        dead_cnt = int(np.sum(self.usage_ema < 0.05 * u_star))
        ent = round(self.usage_entropy(), 4)
        return {
            "architecture": "Non-Transformer Yin-Yang DAO Core (A20 Conscience Homeostasis)",
            "n_prototypes": self.cfg.n_prototypes,
            "num_prototypes": self.cfg.n_prototypes,
            "dim_z": self.cfg.dim_z,
            "top_k": self.cfg.top_k,
            "step_count": self.step_count,
            "sleep_cycles": self.sleep_cycles,
            "split_merge_events": self.split_merge_events,
            "usage_entropy": ent,
            "usage_entropy_ratio": ent,
            "dead_prototypes": dead_cnt,
            "conscience_theta_std": round(float(np.std(self.theta)), 4),
            "yin_mean": round(float(np.mean(self.yin_state)), 4),
            "yang_mean": round(float(np.mean(self.yang_state)), 6),
            "episodic_buffer_size": len(self.episodic_buffer),
            "hippocampal_buffer_size": len(self.episodic_buffer),
            "last_telemetry": self.last_telemetry,
        }

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "prototypes": self.prototypes.tolist(),
            "prototype_labels": list(self.prototype_labels),
            "usage_ema": self.usage_ema.tolist(),
            "theta": self.theta.tolist(),
            "yin_state": self.yin_state.tolist(),
            "yang_state": self.yang_state.tolist(),
            "recurrent_state": self.recurrent_state.tolist(),
            "reward_baseline": self.reward_baseline,
            "step_count": self.step_count,
            "sleep_cycles": self.sleep_cycles,
            "split_merge_events": self.split_merge_events,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        try:
            p = np.asarray(state.get("prototypes", []), dtype=np.float32)
            if p.shape == self.prototypes.shape:
                self.prototypes = p
            labels = state.get("prototype_labels")
            if isinstance(labels, list) and len(labels) == self.cfg.n_prototypes:
                self.prototype_labels = [str(x) for x in labels]
            u = np.asarray(state.get("usage_ema", []), dtype=np.float32)
            if u.shape == self.usage_ema.shape:
                self.usage_ema = u
            th = np.asarray(state.get("theta", []), dtype=np.float32)
            if th.shape == self.theta.shape:
                self.theta = th
            self.reward_baseline = float(state.get("reward_baseline", self.reward_baseline))
            self.step_count = int(state.get("step_count", self.step_count))
            self.sleep_cycles = int(state.get("sleep_cycles", self.sleep_cycles))
            self.split_merge_events = int(state.get("split_merge_events", self.split_merge_events))
        except Exception:
            pass
