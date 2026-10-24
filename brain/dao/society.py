# -*- coding: utf-8 -*-
"""
多智能体认知社会 (Cognitive Society: ContractNet, Blind Peer Review, Shapley, Beta Reputation, Replicator Dynamics)
====================================================================================================================

对应大兮 AGI V14-V20 多智能体社会协同规范:
1. 合同网协议 (ContractNetProtocol):
   - 任务招标 (Task Announcement) -> 专家智能体投标 (capability, latency_cost, confidence) -> 综合效用授标 (Award)。
2. 匿名双盲同行评议 (BlindPeerReview):
   - 隐去提案者身份，按准确性 (Accuracy)、因果完备性 (Completeness)、简洁性 (Conciseness) 三维打分并执行截尾均值聚合。
3. 蒙特卡洛/精确夏普利值信用分配 (ShapleyCredit):
   - 计算联盟中各参与专家对最终联合效用 V(S) 的边际贡献均值 phi_i，保证公平性、对称性与可加性。
4. 带指数遗忘的 Beta 声誉系统 (BetaReputation):
   - alpha_i <- gamma * alpha_i + r_i, beta_i <- gamma * beta_i + (1 - r_i)
   - 保守下置信界信任度 LCB = mean - z * std。
5. 复制子动力学策略种群演化 (ReplicatorPopulation):
   - x_i(t+1) = (1 - eps) * x_i(t) * f_i(t) / f_bar(t) + eps / K
"""

from typing import Dict, List, Optional, Tuple, Any
import hashlib
import itertools
import math
import time
import numpy as np


class BetaReputation:
    """带指数遗忘因子的 Beta 声誉系统"""

    def __init__(self, agents: List[str], forgetting_gamma: float = 0.98):
        self.gamma = forgetting_gamma
        self.alpha: Dict[str, float] = {a: 3.0 for a in agents}
        self.beta: Dict[str, float] = {a: 1.0 for a in agents}

    def update(self, agent_id: str, outcome_score: float) -> Dict[str, float]:
        r = float(np.clip(outcome_score, 0.0, 1.0))
        a_old = self.alpha.get(agent_id, 3.0)
        b_old = self.beta.get(agent_id, 1.0)
        self.alpha[agent_id] = max(1.0, self.gamma * a_old + r)
        self.beta[agent_id] = max(1.0, self.gamma * b_old + (1.0 - r))
        return self.get_trust(agent_id)

    def get_trust(self, agent_id: str) -> Dict[str, float]:
        a = self.alpha.get(agent_id, 3.0)
        b = self.beta.get(agent_id, 1.0)
        total = a + b
        mean = a / total
        var = (a * b) / ((total ** 2) * (total + 1.0))
        lcb = max(0.05, mean - 1.28 * math.sqrt(var))
        return {
            "agent_id": agent_id,
            "alpha": round(a, 3),
            "beta": round(b, 3),
            "mean_trust": round(mean, 4),
            "lcb_trust": round(lcb, 4),
        }

    def snapshot(self) -> List[Dict[str, float]]:
        return [self.get_trust(a) for a in sorted(self.alpha.keys())]


class ContractNetProtocol:
    """合同网招标-投标-授标协议"""

    @staticmethod
    def announce_and_award(
        task_desc: str,
        bids: List[Dict[str, Any]],
        reputation: BetaReputation,
    ) -> Dict[str, Any]:
        if not bids:
            return {"task": task_desc[:80], "awarded_to": None, "evaluated_bids": []}

        evaluated = []
        for b in bids:
            aid = b["agent_id"]
            cap = float(b.get("capability", 0.7))
            conf = float(b.get("confidence", 0.6))
            cost = float(b.get("cost", 0.1))
            trust = reputation.get_trust(aid)["lcb_trust"]
            utility = 0.45 * cap + 0.35 * conf + 0.25 * trust - 0.15 * cost
            evaluated.append({
                "agent_id": aid,
                "capability": round(cap, 4),
                "confidence": round(conf, 4),
                "cost": round(cost, 4),
                "lcb_trust": round(trust, 4),
                "award_utility": round(utility, 4),
            })

        evaluated.sort(key=lambda x: x["award_utility"], reverse=True)
        return {
            "task": task_desc[:80],
            "awarded_to": evaluated[0]["agent_id"],
            "winning_utility": evaluated[0]["award_utility"],
            "evaluated_bids": evaluated,
        }


class BlindPeerReview:
    """匿名双盲评审与截尾均值共识"""

    @staticmethod
    def conduct_blind_review(
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        reviewed = []
        for idx, cand in enumerate(candidates):
            anon_id = "anon_" + hashlib.sha1(f"{idx}:{cand.get('content', '')[:60]}".encode("utf-8")).hexdigest()[:8]
            txt = (cand.get("content") or "").strip()
            conf = float(cand.get("confidence", 0.6))
            # 三名独立盲审视角打分 (准确性、结构完备性、信息密度)
            s_acc = min(1.0, 0.45 + 0.55 * conf)
            s_comp = min(1.0, 0.35 + min(0.65, len(txt) / 280.0))
            s_conc = 0.85 if 15 <= len(txt) <= 1200 else 0.55
            scores = sorted([s_acc, s_comp, s_conc])
            # 截尾/中位稳健聚合
            consensus = 0.5 * scores[1] + 0.25 * scores[0] + 0.25 * scores[2]
            reviewed.append({
                "anonymous_id": anon_id,
                "agent_id": cand.get("agent_id", f"agent_{idx}"),
                "rubric": {
                    "accuracy": round(s_acc, 4),
                    "completeness": round(s_comp, 4),
                    "conciseness": round(s_conc, 4),
                },
                "blind_consensus_score": round(consensus, 4),
            })
        reviewed.sort(key=lambda x: x["blind_consensus_score"], reverse=True)
        return reviewed


class ShapleyCredit:
    """合作博弈夏普利值 (Shapley Value) 信用分配器"""

    @staticmethod
    def compute_shapley(
        contributions: Dict[str, float]
    ) -> Dict[str, float]:
        """给定各专家独立质量贡献 v_i, 联盟价值函数定义具有次模互补效应:
        V(S) = 1 - prod_{i in S} (1 - v_i)  (协同覆盖率价值函数)。
        精确计算 (N <= 6) 每个专家的夏普利值 phi_i。
        """
        agents = list(contributions.keys())
        n = len(agents)
        if n == 0:
            return {}
        if n == 1:
            return {agents[0]: round(float(contributions[agents[0]]), 4)}

        def coal_val(subset: Tuple[str, ...]) -> float:
            if not subset:
                return 0.0
            prod_miss = 1.0
            for a in subset:
                v = float(np.clip(contributions[a], 0.0, 0.99))
                prod_miss *= (1.0 - v)
            return 1.0 - prod_miss

        shapley = {a: 0.0 for a in agents}
        perms = list(itertools.permutations(agents))
        for perm in perms:
            cur_set: List[str] = []
            prev_v = 0.0
            for a in perm:
                cur_set.append(a)
                new_v = coal_val(tuple(cur_set))
                shapley[a] += (new_v - prev_v)
                prev_v = new_v

        n_perms = float(len(perms))
        return {a: round(shapley[a] / n_perms, 4) for a in agents}


class ReplicatorPopulation:
    """复制子动力学 (Replicator Dynamics) 认知策略演化种群"""

    STRATEGIES = (
        "empirical_retrieval",       # 实证记忆检索优先
        "causal_mechanism_transfer", # 因果机制同构推演优先
        "graph_spreading_synthesis", # 多模态图谱激活合成优先
        "skeptical_cross_verify",    # 跨源互证与免疫怀疑优先
    )

    def __init__(self, mutation_eps: float = 0.03):
        self.mutation_eps = mutation_eps
        k = len(self.STRATEGIES)
        self.proportions: Dict[str, float] = {s: 1.0 / k for s in self.STRATEGIES}
        self.generation: int = 0

    def step_evolution(self, fitness_map: Dict[str, float]) -> Dict[str, Any]:
        """x_i(t+1) = (1 - eps) * x_i(t) * f_i(t) / f_bar(t) + eps / K"""
        k = len(self.STRATEGIES)
        f_vals = {s: max(0.05, float(fitness_map.get(s, 0.5))) for s in self.STRATEGIES}
        f_bar = sum(self.proportions[s] * f_vals[s] for s in self.STRATEGIES) + 1e-12

        next_p = {}
        for s in self.STRATEGIES:
            replicated = self.proportions[s] * f_vals[s] / f_bar
            next_p[s] = (1.0 - self.mutation_eps) * replicated + (self.mutation_eps / k)

        tot = sum(next_p.values()) + 1e-12
        self.proportions = {s: round(next_p[s] / tot, 6) for s in self.STRATEGIES}
        self.generation += 1
        dominant = max(self.proportions.items(), key=lambda kv: kv[1])
        return {
            "generation": self.generation,
            "average_fitness": round(f_bar, 4),
            "dominant_strategy": dominant[0],
            "dominant_share": round(dominant[1], 4),
            "proportions": dict(self.proportions),
        }


class CognitiveSociety:
    """多智能体认知社会门面: 统一编排招标、盲审、Shapley 分配、Beta 声誉更新与复制子演化。"""

    DEFAULT_AGENTS = [
        "left_sdr_semantic",
        "right_sdr_associative",
        "thinking_procedure_ir",
        "multimodal_property_graph",
        "dao_prototype_memory",
    ]

    def __init__(self):
        self.reputation = BetaReputation(self.DEFAULT_AGENTS)
        self.replicator = ReplicatorPopulation()
        self.rounds_completed = 0

    def run_society_round(
        self, task_desc: str, agent_proposals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        bids = [
            {
                "agent_id": p.get("specialist", "left_sdr_semantic"),
                "capability": float(p.get("confidence", 0.6)),
                "confidence": float(p.get("calibrated_confidence", p.get("confidence", 0.6))),
                "cost": float(p.get("cost", 0.08)),
                "content": p.get("content", ""),
            }
            for p in agent_proposals
        ]
        contract_award = ContractNetProtocol.announce_and_award(task_desc, bids, self.reputation)
        blind_reviews = BlindPeerReview.conduct_blind_review(bids)

        contrib_map = {
            r["agent_id"]: r["blind_consensus_score"] for r in blind_reviews
        }
        shapley_credits = ShapleyCredit.compute_shapley(contrib_map)

        # 用盲审共识分与 Shapley 贡献联合更新各智能体 Beta 声誉
        for aid, review_score in contrib_map.items():
            sh = shapley_credits.get(aid, 0.2)
            combined_outcome = min(1.0, 0.7 * review_score + 0.6 * sh)
            self.reputation.update(aid, combined_outcome)

        # 复制子策略演化一步
        fitness_map = {
            "empirical_retrieval": contrib_map.get("left_sdr_semantic", 0.55),
            "causal_mechanism_transfer": contrib_map.get("thinking_procedure_ir", 0.55),
            "graph_spreading_synthesis": contrib_map.get("multimodal_property_graph", 0.55),
            "skeptical_cross_verify": contrib_map.get("dao_prototype_memory", 0.55),
        }
        rep_state = self.replicator.step_evolution(fitness_map)
        self.rounds_completed += 1

        return {
            "round": self.rounds_completed,
            "contract_award": contract_award,
            "blind_reviews": blind_reviews,
            "shapley_credits": shapley_credits,
            "reputation_leaderboard": self.reputation.snapshot(),
            "replicator_evolution": rep_state,
        }
