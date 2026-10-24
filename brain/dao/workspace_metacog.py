# -*- coding: utf-8 -*-
"""
全局认知工作区、预测加工惊奇触发、元认知守卫与多模型校准合议
(Global Workspace, Predictive Coding Surprise, Metacognitive Guard & MME Council)
=================================================================================

对应大兮 AGI V14-V20 核心规范:
1. 全局认知工作区 (GlobalCognitiveWorkspace - GWT):
   - 多路专门化认知模块 (Left-SDR 语义路、Right-SDR 联想路、ThinkingIndex 思维链、TypedPropertyGraph 图谱、DAO 原型记忆)
     提交候选联盟 (WorkspaceCoalition)。
   - 容量受限竞争点火 (Capacity-Limited Competitive Ignition, C_max=3) + 冗余多样性惩罚 + 全脑广播记录。

2. 预测加工与主动好奇惊奇度 (PredictiveCodingSurprise):
   - 综合空间表征惊奇 (1 - max_sim)、认识论分歧惊奇 (MME 合议 JS 散度) 与图谱结构缺口惊奇。
   - 当 S_t >= tau_surprise 时触发主动求知 (Active Curiosity)，生成针对性知识缺口查询推入自主学习队列。

3. 元认知 EVC 路由器与 Lyapunov 收敛守卫 (MetacognitiveGuard):
   - 期望计算价值 EVC = Delta E[Utility] - Cost_compute，动态决策走 System-1 快速检索、System-2 深度 DAO 推演、
     或触发外部多源自主采集。
   - 追踪 Lyapunov 能量函数 V(s_t) = ||r_t||^2 + lambda_n * ||n_t||^2，保证迭代推演单调收敛不发散。

4. 多模型校准合议 (MultiModelCalibratedCouncil - MME):
   - 温度缩放 + 保序校准 (Temperature Scaling & Isotonic Calibration)
   - 修剪对数池化 (Trimmed Log-Linear Pooling) + 认识论不确定性门控
   - 汤普森采样上下文路由 (Thompson Sampling) + 在线指数加权专家更新 (Hedge)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import math
import time
import numpy as np


@dataclass
class WorkspaceCoalition:
    specialist_id: str
    modality: str
    topic: str
    content: str
    raw_confidence: float
    salience: float
    evidence_source: str = "memory"
    meta: Dict[str, Any] = field(default_factory=dict)


class GlobalCognitiveWorkspace:
    """全局认知工作区 (GWT): 容量受限竞争点火与全脑广播。"""

    def __init__(self, capacity: int = 3, diversity_penalty: float = 0.25):
        self.capacity = max(1, capacity)
        self.diversity_penalty = diversity_penalty
        self.broadcast_history: List[Dict[str, Any]] = []

    @staticmethod
    def _token_overlap(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        sa = set(a[:120])
        sb = set(b[:120])
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / float(len(sa | sb))

    def compete_and_ignite(
        self, coalitions: List[WorkspaceCoalition]
    ) -> Dict[str, Any]:
        if not coalitions:
            return {"ignited": [], "suppressed": [], "winning_specialist": None}

        scored = []
        for c in coalitions:
            base_score = 0.6 * float(c.raw_confidence) + 0.4 * float(c.salience)
            scored.append((base_score, c))
        scored.sort(key=lambda x: x[0], reverse=True)

        ignited: List[Tuple[float, WorkspaceCoalition]] = []
        suppressed: List[Dict[str, Any]] = []

        for base_score, cand in scored:
            if len(ignited) >= self.capacity:
                suppressed.append({
                    "specialist_id": cand.specialist_id,
                    "topic": cand.topic,
                    "score": round(base_score, 4),
                    "reason": "capacity_limit",
                })
                continue
            # 计算与已点火联盟的冗余惩罚
            max_ov = max(
                (self._token_overlap(cand.content, prev.content) for _, prev in ignited),
                default=0.0,
            )
            eff_score = base_score - self.diversity_penalty * max_ov
            if eff_score >= 0.15 or not ignited:
                ignited.append((eff_score, cand))
            else:
                suppressed.append({
                    "specialist_id": cand.specialist_id,
                    "topic": cand.topic,
                    "score": round(eff_score, 4),
                    "reason": "redundancy_suppressed",
                })

        ignited.sort(key=lambda x: x[0], reverse=True)
        winner = ignited[0][1] if ignited else None
        broadcast_event = {
            "ts": time.time(),
            "winning_specialist": winner.specialist_id if winner else None,
            "winning_topic": winner.topic if winner else "",
            "ignited": [
                {
                    "specialist_id": c.specialist_id,
                    "modality": c.modality,
                    "topic": c.topic,
                    "ignition_score": round(sc, 4),
                    "confidence": round(c.raw_confidence, 4),
                    "evidence_source": c.evidence_source,
                    "preview": (c.content or "")[:120],
                }
                for sc, c in ignited
            ],
            "suppressed": suppressed,
        }
        self.broadcast_history.append(broadcast_event)
        if len(self.broadcast_history) > 50:
            self.broadcast_history.pop(0)
        return broadcast_event


class PredictiveCodingSurprise:
    """预测加工惊奇度评估器与主动好奇种子生成器"""

    def __init__(self, surprise_threshold: float = 0.52):
        self.surprise_threshold = surprise_threshold
        self.surprise_ema: float = 0.25
        self.total_surprises: int = 0

    def evaluate(
        self,
        query: str,
        max_similarity: float,
        epistemic_uncertainty: float = 0.0,
        graph_coverage: int = 0,
    ) -> Dict[str, Any]:
        spatial_surprise = float(np.clip(1.0 - max_similarity, 0.0, 1.0))
        structural_surprise = 1.0 if graph_coverage == 0 else (0.4 if graph_coverage < 2 else 0.05)
        epistemic_s = float(np.clip(epistemic_uncertainty, 0.0, 1.0))

        total_surprise = float(np.clip(
            0.55 * spatial_surprise + 0.25 * epistemic_s + 0.20 * structural_surprise,
            0.0,
            1.0,
        ))
        self.surprise_ema = 0.90 * self.surprise_ema + 0.10 * total_surprise
        is_surprised = total_surprise >= self.surprise_threshold
        if is_surprised:
            self.total_surprises += 1

        # 提炼主动求知探索主题
        curiosity_topic = (query or "").strip()
        for suffix in ("是什么", "怎么回事", "为什么", "如何", "怎么", "有哪些", "请问", "？", "?"):
            curiosity_topic = curiosity_topic.replace(suffix, "")
        curiosity_topic = curiosity_topic.strip()[:40]

        return {
            "total_surprise": round(total_surprise, 4),
            "spatial_surprise": round(spatial_surprise, 4),
            "epistemic_surprise": round(epistemic_s, 4),
            "structural_surprise": round(structural_surprise, 4),
            "surprise_ema": round(self.surprise_ema, 4),
            "threshold": self.surprise_threshold,
            "is_surprised": is_surprised,
            "curiosity_topic": curiosity_topic if is_surprised else None,
        }


class MetacognitiveGuard:
    """元认知 EVC 计算价值决策器与 Lyapunov 能量单调收敛守卫"""

    def __init__(self):
        self.energy_trace: List[float] = []
        self.halt_count: int = 0

    def compute_evc_route(
        self,
        fast_confidence: float,
        surprise_score: float,
        is_procedural_query: bool = False,
    ) -> Dict[str, Any]:
        """根据期望计算价值 EVC = Delta E[U] - C_compute 选择认知路径:
        - system1_fast_recall: 高置信低惊奇事实直接毫秒级响应
        - system2_dao_deliberation: 复杂任务或中等惊奇启动阴阳双态深度联想+思维链+图谱合成
        - autonomous_web_acquisition: 知识盲区 (高惊奇低置信) 触发多源主动采集与跨源互证
        """
        c_sys2 = 0.12
        c_web = 0.28
        delta_u_sys2 = max(0.0, (1.0 - fast_confidence) * 0.65 + (0.25 if is_procedural_query else 0.0))
        delta_u_web = max(0.0, surprise_score * 0.85 - fast_confidence * 0.40)

        evc_sys2 = delta_u_sys2 - c_sys2
        evc_web = delta_u_web - c_web

        if fast_confidence >= 0.72 and surprise_score < 0.45 and not is_procedural_query:
            route = "system1_fast_recall"
        elif evc_web > evc_sys2 and surprise_score >= 0.58 and fast_confidence < 0.45:
            route = "autonomous_web_acquisition"
        else:
            route = "system2_dao_deliberation"

        return {
            "selected_route": route,
            "evc_system2": round(evc_sys2, 4),
            "evc_web_acquisition": round(evc_web, 4),
            "fast_confidence": round(fast_confidence, 4),
            "surprise_score": round(surprise_score, 4),
        }

    def check_lyapunov(self, prediction_error: float, yang_mean: float) -> Dict[str, Any]:
        """Lyapunov 能量函数 V(s_t) = r_t^2 + 0.5 * yang_t^2 收敛性检查。"""
        v_t = float(prediction_error ** 2 + 0.5 * (yang_mean ** 2))
        prev_v = self.energy_trace[-1] if self.energy_trace else v_t
        delta_v = v_t - prev_v
        self.energy_trace.append(v_t)
        if len(self.energy_trace) > 64:
            self.energy_trace.pop(0)

        # 若连续多步能量上升超过阈值, 触发制动早停
        diverging = False
        if len(self.energy_trace) >= 3:
            if self.energy_trace[-1] > self.energy_trace[-2] > self.energy_trace[-3] and delta_v > 0.15:
                diverging = True
                self.halt_count += 1
        return {
            "lyapunov_v": round(v_t, 6),
            "delta_v": round(delta_v, 6),
            "monotone_non_increasing": delta_v <= 1e-5,
            "diverging_brake": diverging,
        }


class MultiModelCalibratedCouncil:
    """多模型校准合议庭 (MME): 温度缩放校准、修剪对数池化、汤普森采样与 Hedge 在线权重。"""

    SPECIALISTS = (
        "left_sdr_semantic",
        "right_sdr_associative",
        "thinking_procedure_ir",
        "multimodal_property_graph",
        "dao_prototype_memory",
    )

    def __init__(self, trim_ratio: float = 0.20, hedge_eta: float = 0.15):
        self.trim_ratio = trim_ratio
        self.hedge_eta = hedge_eta
        self.temperatures: Dict[str, float] = {s: 1.0 for s in self.SPECIALISTS}
        self.hedge_weights: Dict[str, float] = {s: 1.0 / len(self.SPECIALISTS) for s in self.SPECIALISTS}
        self.ts_alpha: Dict[str, float] = {s: 2.0 for s in self.SPECIALISTS}
        self.ts_beta: Dict[str, float] = {s: 1.0 for s in self.SPECIALISTS}

    def calibrate_confidence(self, specialist: str, raw_conf: float) -> float:
        """温度缩放 + Sigmoid 保序校准"""
        p = float(np.clip(raw_conf, 1e-4, 1.0 - 1e-4))
        logit = math.log(p / (1.0 - p))
        temp = max(0.25, self.temperatures.get(specialist, 1.0))
        cal_logit = logit / temp
        cal_p = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, cal_logit))))
        return float(np.clip(cal_p, 0.01, 0.99))

    def deliberate(
        self, proposals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """对多个专家的提案进行校准、修剪对数池化 (Trimmed Log-Linear Pooling) 并计算认识论不确定性。"""
        if not proposals:
            return {
                "pooled_confidence": 0.0,
                "epistemic_uncertainty": 1.0,
                "abstain": True,
                "selected_proposal": None,
                "calibrated_proposals": [],
            }

        calibrated = []
        for prop in proposals:
            sp = prop.get("specialist", "left_sdr_semantic")
            raw_c = float(prop.get("confidence", 0.5))
            cal_c = self.calibrate_confidence(sp, raw_c)
            hw = self.hedge_weights.get(sp, 0.2)
            calibrated.append({
                **prop,
                "calibrated_confidence": round(cal_c, 4),
                "hedge_weight": round(hw, 4),
                "combined_score": round(cal_c * (0.7 + 0.3 * hw * len(self.SPECIALISTS)), 4),
            })

        calibrated.sort(key=lambda x: x["combined_score"], reverse=True)
        confs = [c["calibrated_confidence"] for c in calibrated]

        # 修剪两端极端值后执行对数线性池化 (Trimmed Log-Linear Pooling)
        if len(confs) >= 4:
            trimmed = sorted(confs)[1:-1]
        else:
            trimmed = confs

        log_pos = sum(math.log(max(1e-6, p)) for p in trimmed) / len(trimmed)
        log_neg = sum(math.log(max(1e-6, 1.0 - p)) for p in trimmed) / len(trimmed)
        max_lp = max(log_pos, log_neg)
        pooled = math.exp(log_pos - max_lp) / (math.exp(log_pos - max_lp) + math.exp(log_neg - max_lp))

        # 认识论不确定性 (专家间方差/分歧度)
        epistemic_u = float(np.std(confs)) * 2.0 if len(confs) > 1 else (1.0 - pooled) * 0.5
        epistemic_u = float(np.clip(epistemic_u, 0.0, 1.0))

        best = calibrated[0]
        abstain = pooled < 0.22 and epistemic_u > 0.65

        return {
            "pooled_confidence": round(float(pooled), 4),
            "epistemic_uncertainty": round(epistemic_u, 4),
            "abstain": abstain,
            "selected_proposal": best,
            "calibrated_proposals": calibrated,
        }

    def update_online(self, specialist: str, reward: float) -> None:
        """Hedge 在线指数加权更新 + Thompson Sampling 后验更新"""
        if specialist not in self.hedge_weights:
            self.hedge_weights[specialist] = 0.2
            self.ts_alpha[specialist] = 2.0
            self.ts_beta[specialist] = 1.0
        r = float(np.clip(reward, 0.0, 1.0))
        loss = 1.0 - r
        self.hedge_weights[specialist] *= math.exp(-self.hedge_eta * loss)
        total_w = sum(self.hedge_weights.values()) + 1e-12
        for k in self.hedge_weights:
            self.hedge_weights[k] /= total_w

        self.ts_alpha[specialist] += r
        self.ts_beta[specialist] += (1.0 - r)

    def thompson_sample_ranking(self, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        rng = np.random.RandomState(seed if seed is not None else int(time.time() * 1000) % 100000)
        out = []
        for sp in self.SPECIALISTS:
            a = self.ts_alpha.get(sp, 2.0)
            b = self.ts_beta.get(sp, 1.0)
            sample = float(rng.beta(a, b))
            out.append({
                "specialist": sp,
                "thompson_sample": round(sample, 4),
                "posterior_mean": round(a / (a + b), 4),
                "hedge_weight": round(self.hedge_weights.get(sp, 0.2), 4),
            })
        out.sort(key=lambda x: x["thompson_sample"], reverse=True)
        return out
