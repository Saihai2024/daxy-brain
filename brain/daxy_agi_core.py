# -*- coding: utf-8 -*-
"""
大兮 AGI 统一认知中枢 (DaxyAgiEngine)
======================================

将全部 19 份技术文档中的大兮 AGI 自主创新算法体系无缝集成到 `daxy-brain`:
  1. 非 Transformer 阴阳双态 DAO 认知核心 (`DaoCognitiveCore` + A20 良心稳态稀疏原型路由)
  2. 三因子局部可塑性与双系统海马体-皮层睡眠重放巩固 (`sleep_consolidate`)
  3. 一次经验机制获取 (`MechanismExtractor`)、`ProcedureIR` 状态机、跨领域结构同构迁移 (`StructuralTransferEngine`)
     与反例边界收紧 (`CounterexampleRefiner`)
  4. 万物皆可图多模态类型属性图 (`TypedPropertyGraph`)、确定性哈希、增量差分与多跳激活扩散
  5. 语义拓展需求合同 (`SemanticContract`) 与局部最小修复 DAG (`MinimalRepairDAG`)
  6. 全局认知工作区 (`GlobalCognitiveWorkspace`)、预测加工主动惊奇 (`PredictiveCodingSurprise`)、
     元认知 EVC 路由与 Lyapunov 收敛守卫 (`MetacognitiveGuard`)
  7. 多模型校准合议庭 (`MultiModelCalibratedCouncil` - MME) 与多智能体社会 (`CognitiveSociety`:
     ContractNet 招标授标、匿名双盲评审、Shapley 夏普利信用分配、Beta 声誉、复制子策略种群演化)
  8. 惊奇驱动自主多源采集、跨源互证投票、认知免疫反污染与 V20 八大行动 SHA-256 证据账本 (`AutonomousLearningLoop`)
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import threading
import time
import numpy as np

from brain.dao import (
    DaoConfig,
    DaoCognitiveCore,
    MechanismExtractor,
    StructuralTransferEngine,
    CounterexampleRefiner,
    ProcedureIR,
    TypedPropertyGraph,
    SemanticContract,
    MinimalRepairDAG,
    WorkspaceCoalition,
    GlobalCognitiveWorkspace,
    PredictiveCodingSurprise,
    MetacognitiveGuard,
    MultiModelCalibratedCouncil,
    CognitiveSociety,
    V20EvidenceLedger,
    AutonomousLearningLoop,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGI_STATE_PATH = ROOT / "brain" / "models" / "daxy_agi_state.json"


class DaxyAgiEngine:
    """大兮 AGI 核心认知引擎门面"""

    def __init__(self, brain: Optional[Any] = None, state_path: Optional[Path] = None):
        self.brain = brain
        self.state_path = Path(state_path or DEFAULT_AGI_STATE_PATH)
        self._lock = threading.RLock()

        # 八大核心子系统初始化
        self.dao_core = DaoCognitiveCore(DaoConfig())
        self.extractor = MechanismExtractor()
        self.transfer_engine = StructuralTransferEngine()
        self.refiner = CounterexampleRefiner()
        self.graph = TypedPropertyGraph()
        self.workspace = GlobalCognitiveWorkspace(capacity=3)
        self.surprise_evaluator = PredictiveCodingSurprise(surprise_threshold=0.52)
        self.metacog = MetacognitiveGuard()
        self.council = MultiModelCalibratedCouncil()
        self.society = CognitiveSociety()
        self.ledger = V20EvidenceLedger()
        self.learner = AutonomousLearningLoop(self.ledger)

        # 已注册的 ProcedureIR 机制库
        self.procedures: Dict[str, ProcedureIR] = {}

        # 加载已有持久化状态并做基础认知种子热身
        self.load_state()
        self._bootstrap_seed_knowledge()

    def attach_brain(self, brain: Any) -> None:
        self.brain = brain

    def _bootstrap_seed_knowledge(self) -> None:
        """初始化核心架构自举图谱与基础因果机制，确保零冷启动空白。"""
        if len(self.graph.nodes) == 0:
            seed_passages = [
                (
                    "大兮认知大脑",
                    "大兮认知大脑是一种非Transformer架构的自主演进认知智能系统。大兮认知大脑基于阴阳双态动力学与稀疏原型路由，大兮认知大脑导致零自注意力二次方开销。",
                    "core_spec",
                    0.96,
                ),
                (
                    "阴阳双态动力学",
                    "阴阳双态动力学属于大兮认知大脑的核心部分。阴态激活负责兴奋性联想记忆整合，阳态抑制基于预测误差驱动制动门控，阳态抑制导致零误差时零误伤率。",
                    "core_spec",
                    0.95,
                ),
                (
                    "良心稳态路由",
                    "良心稳态路由是一种局部稀疏原型路由算法。良心稳态路由依赖于使用率指数滑动平均与偏置反馈，良心稳态路由导致原型表征空间高信息熵防坍缩。",
                    "core_spec",
                    0.95,
                ),
                (
                    "三因子局部可塑性",
                    "三因子局部可塑性是一种生物启发突触学习法则。三因子局部可塑性依赖于前突触路由、后突触阴态激活与多巴胺惊奇调质信号，驱动海马体到皮层的睡眠巩固。",
                    "core_spec",
                    0.95,
                ),
            ]
            for topic, text, src, cred in seed_passages:
                self.graph.ingest_knowledge_passage(topic, text, source=src, credibility=cred)
            self.ledger.record(
                "A4_multimodal_typed_property_graph",
                "bootstrap_core_graph",
                {"nodes": len(self.graph.nodes), "edges": len(self.graph.edges), "hash": self.graph.graph_hash()},
            )

        if not self.procedures:
            ir = self.extractor.extract_one_shot(
                title="数据库大表不停机 Schema 迁移机制",
                experience_text=(
                    "1. 预检源表主从延迟与锁等待水位；"
                    "2. 创建影子表并挂载增量双写触发器；"
                    "3. 分批低优回填历史存量数据并执行行级哈希对账；"
                    "4. 原子切换读流量至新表并保留逆向回滚通道。"
                ),
                source_domain="database_reliability",
            )
            self.register_procedure(ir)

    def register_procedure(self, proc_ir: ProcedureIR) -> None:
        with self._lock:
            self.procedures[proc_ir.ir_id] = proc_ir
            self.graph.upsert_node(
                label=proc_ir.signature.name,
                node_type="procedure_ir",
                modality="procedure",
                credibility=0.90,
                properties={"ir_id": proc_ir.ir_id, "domain": proc_ir.signature.source_domain, "digest": proc_ir.digest()},
            )

    def observe_query_fast(
        self,
        query: str,
        query_vec: Optional[np.ndarray],
        max_sim: float,
        matched_topic: Optional[str] = None,
        hemisphere: str = "left",
    ) -> Dict[str, Any]:
        """在 CognitiveBrain.answer() 热路径中轻量调用 (<0.5ms):
        更新 DAO 阴阳双态路由、计算预测惊奇度、并在高惊奇时将未知主题推入自主学习队列。
        """
        with self._lock:
            pred_error = float(max(0.0, 1.0 - max_sim))
            if query_vec is None:
                # 确定性字符哈希向量兜底
                arr = np.frombuffer((query or " ").encode("utf-8"), dtype=np.uint8).astype(np.float32)
                query_vec = arr
            dao_out = self.dao_core.step(
                query_vec,
                prediction_error=pred_error,
                learn=(max_sim >= 0.48),
                topic_label=matched_topic or query[:32],
            )
            graph_hits = self.graph.spread_activation([matched_topic or query[:16]], hops=1, top_n=4)
            surprise_info = self.surprise_evaluator.evaluate(
                query=query,
                max_similarity=max_sim,
                epistemic_uncertainty=pred_error * 0.5,
                graph_coverage=len(graph_hits),
            )
            is_proc = any(k in (query or "") for k in ("如何", "怎么", "步骤", "方案", "排查", "迁移", "实现", "修复"))
            evc_info = self.metacog.compute_evc_route(
                fast_confidence=max_sim,
                surprise_score=surprise_info["total_surprise"],
                is_procedural_query=is_proc,
            )
            lyapunov_info = self.metacog.check_lyapunov(
                prediction_error=pred_error,
                yang_mean=dao_out["telemetry"]["yang_mean"],
            )

            if surprise_info["is_surprised"] and surprise_info.get("curiosity_topic"):
                self.learner.enqueue_surprise_gap(
                    topic=surprise_info["curiosity_topic"],
                    raw_query=query,
                    surprise_score=surprise_info["total_surprise"],
                    persist_pg=False,  # 热路径不阻塞写库, 由后台或异步刷入
                )

            return {
                "dao": dao_out["telemetry"],
                "surprise": surprise_info,
                "evc": evc_info,
                "lyapunov": lyapunov_info,
                "hemisphere": hemisphere,
                "graph_activated": len(graph_hits),
            }

    def deliberate(
        self,
        query: str,
        thinking_index: Optional[Any] = None,
        auto_acquire_on_surprise: bool = False,
    ) -> Dict[str, Any]:
        """执行完整的大兮 AGI 全链路深度认知推演:
        1. 编译语义拓展合同 (`SemanticContract`)
        2. 收集五路专门化认知专家提案 (Left-SDR / Right-SDR / Thinking-ProcedureIR / Multimodal-Graph / DAO-Prototype)
        3. 全局工作区竞争点火与全脑广播 (`GlobalCognitiveWorkspace`)
        4. DAO 阴阳双态循环与良心稳态路由 (`DaoCognitiveCore`)
        5. MME 多模型校准合议 (`MultiModelCalibratedCouncil`) + 多智能体社会盲审与 Shapley 分配 (`CognitiveSociety`)
        6. 预测惊奇度与 EVC/Lyapunov 守卫评估 (`PredictiveCodingSurprise` & `MetacognitiveGuard`)
        7. 局部最小修复 DAG 校验与修补 (`MinimalRepairDAG`)
        8. 记录 V20 八大行动证据链
        """
        t0 = time.time()
        q = (query or "").strip()
        with self._lock:
            # 1. 语义拓展合同
            contract = SemanticContract.compile_from_query(q)

            # 2. 收集多路认知专家提案
            coalitions: List[WorkspaceCoalition] = []
            proposals: List[Dict[str, Any]] = []
            max_sim = 0.0
            key_vec = None

            if self.brain is not None:
                try:
                    key_vec = self.brain.encode(q)
                    ans_text, dbg = self.brain.answer(q)
                    max_sim = float(dbg.get("similarity") or dbg.get("best_sim") or 0.0)
                    hem = dbg.get("hemisphere", "left")
                    if ans_text:
                        sp_id = "left_sdr_semantic" if hem == "left" else "right_sdr_associative"
                        coalitions.append(WorkspaceCoalition(
                            specialist_id=sp_id,
                            modality="text",
                            topic=dbg.get("topic", q[:32]),
                            content=ans_text,
                            raw_confidence=min(0.95, max(0.35, max_sim)),
                            salience=0.85,
                            evidence_source=dbg.get("source", "cognitive_brain"),
                        ))
                        proposals.append({
                            "specialist": sp_id,
                            "topic": dbg.get("topic", q[:32]),
                            "content": ans_text,
                            "confidence": min(0.95, max(0.35, max_sim)),
                            "cost": 0.04,
                        })
                    # 调取 top-3 候选丰富右脑联想提案
                    top_cands = self.brain.retrieve(q, top_k=3)
                    if len(top_cands) >= 2:
                        m2, sim2, sc2 = top_cands[1]
                        coalitions.append(WorkspaceCoalition(
                            specialist_id="right_sdr_associative",
                            modality="text",
                            topic=m2["topic"],
                            content=m2["content"],
                            raw_confidence=float(np.clip(sim2, 0.20, 0.90)),
                            salience=0.68,
                            evidence_source=m2.get("source", "associative"),
                        ))
                        proposals.append({
                            "specialist": "right_sdr_associative",
                            "topic": m2["topic"],
                            "content": m2["content"],
                            "confidence": float(np.clip(sim2, 0.20, 0.90)),
                            "cost": 0.05,
                        })
                except Exception:
                    pass

            # 思维链 / ProcedureIR 专家路
            if thinking_index is not None:
                try:
                    th_ans, th_dbg = thinking_index.answer(q)
                    if th_ans:
                        rel = float(th_dbg.get("rel", 0.65))
                        coalitions.append(WorkspaceCoalition(
                            specialist_id="thinking_procedure_ir",
                            modality="procedure",
                            topic=th_dbg.get("matched_task", q[:40]),
                            content=th_ans,
                            raw_confidence=min(0.95, max(0.45, rel)),
                            salience=0.90,
                            evidence_source="thinking_corpus",
                        ))
                        proposals.append({
                            "specialist": "thinking_procedure_ir",
                            "topic=matched": th_dbg.get("matched_task", q[:40]),
                            "content": th_ans,
                            "confidence": min(0.95, max(0.45, rel)),
                            "cost": 0.08,
                        })
                except Exception:
                    pass

            # 内置 ProcedureIR 机制库匹配
            for ir in self.procedures.values():
                if ir.signature.name[:6] in q or any(tok in q for tok in ir.signature.name.split()[:2] if len(tok) >= 2):
                    steps_txt = "；".join(f"({s.step_no}) {s.action} [{s.tool}]" for s in ir.steps[:5])
                    ir_content = f"【可执行机制 {ir.signature.name}】前置条件：{'，'.join(ir.signature.preconditions)}。执行步骤：{steps_txt}。"
                    coalitions.append(WorkspaceCoalition(
                        specialist_id="thinking_procedure_ir",
                        modality="procedure",
                        topic=ir.signature.name,
                        content=ir_content,
                        raw_confidence=0.84,
                        salience=0.88,
                        evidence_source=f"procedure_ir:{ir.ir_id}",
                    ))
                    proposals.append({
                        "specialist": "thinking_procedure_ir",
                        "topic": ir.signature.name,
                        "content": ir_content,
                        "confidence": 0.84,
                        "cost": 0.06,
                    })
                    break

            # 多模态类型属性图谱激活扩散路
            seed_terms = [q.replace("是什么", "").replace("？", "").replace("?", "").strip()[:24]]
            graph_spread = self.graph.spread_activation(seed_terms, hops=2, top_n=6)
            if graph_spread:
                traces = [g["trace"] for g in graph_spread if g.get("trace") and g["trace"] != "seed"]
                previews = [
                    g["properties"].get("preview", "")
                    for g in graph_spread
                    if isinstance(g.get("properties"), dict) and g["properties"].get("preview")
                ]
                g_text_parts = []
                if previews:
                    g_text_parts.append(previews[0])
                if traces:
                    g_text_parts.append("图谱因果链：" + "；".join(traces[:3]))
                g_content = " ".join(g_text_parts).strip() or (
                    "关联概念：" + "、".join(g["label"] for g in graph_spread[:5])
                )
                g_conf = float(graph_spread[0]["activation"]) * 0.85
                coalitions.append(WorkspaceCoalition(
                    specialist_id="multimodal_property_graph",
                    modality="graph",
                    topic=graph_spread[0]["label"],
                    content=g_content,
                    raw_confidence=g_conf,
                    salience=0.78,
                    evidence_source=f"graph:{self.graph.graph_hash()}",
                ))
                proposals.append({
                    "specialist": "multimodal_property_graph",
                    "topic": graph_spread[0]["label"],
                    "content": g_content,
                    "confidence": g_conf,
                    "cost": 0.05,
                })

            # 3. DAO 阴阳双态认知步进
            if key_vec is None:
                key_vec = np.frombuffer((q or " ").encode("utf-8"), dtype=np.uint8).astype(np.float32)
            pred_err = float(max(0.0, 1.0 - max(max_sim, 0.7 if proposals else 0.1)))
            dao_res = self.dao_core.step(key_vec, prediction_error=pred_err, learn=True, topic_label=q[:32])
            self.ledger.record("A1_non_transformer_dao_routing", q, dao_res["telemetry"])

            # DAO 原型专家提案
            active_lbls = dao_res["telemetry"]["active_labels"]
            dao_summary = (
                f"DAO 稀疏原型激活集 {active_lbls} "
                f"(阴态整合={dao_res['telemetry']['yin_mean']}, "
                f"阳态门控={dao_res['telemetry']['yang_gate']}, "
                f"良心熵={dao_res['telemetry']['usage_entropy']})"
            )
            proposals.append({
                "specialist": "dao_prototype_memory",
                "topic": active_lbls[0] if active_lbls else "dao_proto",
                "content": proposals[0]["content"] if proposals else dao_summary,
                "confidence": float(dao_res["telemetry"]["yang_gate"]) * 0.75,
                "cost": 0.03,
            })

            # 4. 全局工作区竞争点火
            gw_broadcast = self.workspace.compete_and_ignite(coalitions)

            # 5. MME 多模型校准合议与多智能体社会协同
            mme_result = self.council.deliberate(proposals)
            society_result = self.society.run_society_round(q, mme_result["calibrated_proposals"])
            self.ledger.record(
                "A7_mme_council_cognitive_society",
                q,
                {
                    "pooled_confidence": mme_result["pooled_confidence"],
                    "epistemic_uncertainty": mme_result["epistemic_uncertainty"],
                    "awarded_to": society_result["contract_award"]["awarded_to"],
                    "dominant_strategy": society_result["replicator_evolution"]["dominant_strategy"],
                },
            )

            # 6. 预测加工惊奇度与 EVC / Lyapunov 守卫
            surprise_res = self.surprise_evaluator.evaluate(
                query=q,
                max_similarity=max(max_sim, mme_result["pooled_confidence"]),
                epistemic_uncertainty=mme_result["epistemic_uncertainty"],
                graph_coverage=len(graph_spread),
            )
            evc_res = self.metacog.compute_evc_route(
                fast_confidence=mme_result["pooled_confidence"],
                surprise_score=surprise_res["total_surprise"],
                is_procedural_query=any(k in q for k in ("如何", "怎么", "步骤", "方案", "迁移", "排查")),
            )
            lyapunov_res = self.metacog.check_lyapunov(
                prediction_error=pred_err,
                yang_mean=dao_res["telemetry"]["yang_mean"],
            )
            self.ledger.record(
                "A6_global_workspace_evc_metacog",
                q,
                {
                    "route": evc_res["selected_route"],
                    "surprise": surprise_res["total_surprise"],
                    "lyapunov_v": lyapunov_res["lyapunov_v"],
                },
            )

            # 若开启了高惊奇自动在线采集且当前置信度不足，立即触发多源自主学习
            acquired_info = None
            if surprise_res["is_surprised"] and surprise_res.get("curiosity_topic"):
                self.learner.enqueue_surprise_gap(
                    topic=surprise_res["curiosity_topic"],
                    raw_query=q,
                    surprise_score=surprise_res["total_surprise"],
                    persist_pg=True,
                )
                if auto_acquire_on_surprise and mme_result["pooled_confidence"] < 0.60:
                    acquired_info = self.learner.acquire_and_learn(
                        topic=surprise_res["curiosity_topic"],
                        agi_engine=self,
                        allow_network=True,
                        persist_pg=True,
                    )

            # 选出候选响应并执行语义合同验证与最小修复 DAG
            raw_candidate = ""
            if acquired_info and acquired_info.get("ok"):
                raw_candidate = acquired_info.get("content_preview", "")
            elif mme_result.get("selected_proposal"):
                raw_candidate = mme_result["selected_proposal"].get("content", "")

            repair_res = MinimalRepairDAG.verify_and_repair(
                contract=contract,
                candidate_response=raw_candidate,
                graph_context=graph_spread,
            )
            self.ledger.record(
                "A5_semantic_contract_repair_dag",
                q,
                {
                    "contract_id": contract.contract_id,
                    "passed_initially": repair_res["passed_initially"],
                    "repair_steps": len(repair_res["repair_dag"]),
                    "verified_after_repair": repair_res["verified_after_repair"],
                },
            )

            # 在线更新胜出专家的校准合议权重
            winning_sp = society_result["contract_award"]["awarded_to"]
            if winning_sp:
                self.council.update_online(
                    winning_sp,
                    reward=float(mme_result["pooled_confidence"]),
                )

            elapsed_ms = round((time.time() - t0) * 1000.0, 2)
            return {
                "query": q,
                "final_response": repair_res["repaired_response"],
                "elapsed_ms": elapsed_ms,
                "semantic_contract": {
                    "contract_id": contract.contract_id,
                    "goals": contract.goals,
                    "hard_constraints": contract.hard_constraints,
                    "acceptance_tests": contract.acceptance_tests,
                },
                "repair_dag": repair_res,
                "global_workspace": gw_broadcast,
                "dao_core": dao_res["telemetry"],
                "mme_council": {
                    "pooled_confidence": mme_result["pooled_confidence"],
                    "epistemic_uncertainty": mme_result["epistemic_uncertainty"],
                    "abstain": mme_result["abstain"],
                    "thompson_ranking": self.council.thompson_sample_ranking(seed=42),
                },
                "cognitive_society": society_result,
                "surprise": surprise_res,
                "metacog_evc": evc_res,
                "lyapunov_guard": lyapunov_res,
                "multimodal_graph_spread": graph_spread,
                "autonomous_acquisition": acquired_info,
            }

    def one_shot_learn_and_transfer(
        self,
        title: str,
        experience_text: str,
        source_domain: str = "source_domain",
        target_domain: Optional[str] = None,
        role_overrides: Optional[Dict[str, str]] = None,
        counterexample: Optional[str] = None,
    ) -> Dict[str, Any]:
        """单样本机制获取 + 跨领域同构迁移 + 反例边界收紧端到端闭环。"""
        with self._lock:
            base_ir = self.extractor.extract_one_shot(
                title=title,
                experience_text=experience_text,
                source_domain=source_domain,
            )
            self.register_procedure(base_ir)
            self.graph.ingest_knowledge_passage(
                topic=title,
                content=experience_text,
                source=f"one_shot:{source_domain}",
                credibility=0.92,
            )

            transferred_ir = None
            if target_domain:
                transferred_ir = self.transfer_engine.transfer(
                    source_ir=base_ir,
                    target_domain=target_domain,
                    role_overrides=role_overrides,
                )
                self.register_procedure(transferred_ir)

            active_ir = transferred_ir or base_ir
            if counterexample:
                active_ir = self.refiner.refine(
                    proc_ir=active_ir,
                    counterexample_desc=counterexample,
                    violated_step_no=1,
                )
                self.register_procedure(active_ir)

            ev = self.ledger.record(
                "A3_one_shot_mechanism_transfer",
                title,
                {
                    "base_ir_id": base_ir.ir_id,
                    "transferred_ir_id": transferred_ir.ir_id if transferred_ir else None,
                    "target_domain": target_domain,
                    "negative_guards": len(active_ir.signature.negative_guards),
                    "digest": active_ir.digest(),
                },
            )
            self.save_state()
            return {
                "base_procedure_ir": base_ir.to_dict(),
                "transferred_procedure_ir": transferred_ir.to_dict() if transferred_ir else None,
                "active_procedure_ir": active_ir.to_dict(),
                "evidence_pack": ev,
            }

    def run_sleep_consolidation(self, n_replay: int = 32, lr: float = 0.03) -> Dict[str, Any]:
        """执行 DAO 认知核心三因子睡眠重放巩固并记录 V20 A2 证据。"""
        with self._lock:
            res = self.dao_core.sleep_consolidate(n_replay=n_replay, lr=lr)
            ev = self.ledger.record("A2_three_factor_sleep_consolidation", "sleep_replay", res)
            self.save_state()
            return {**res, "evidence_pack": ev}

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "engine": "DAXY AGI V20 Cognitive Engine (Non-Transformer DAO Architecture)",
                "dao_core": self.dao_core.stats(),
                "multimodal_graph": self.graph.stats(),
                "procedures_count": len(self.procedures),
                "procedures": [p.to_dict() for p in list(self.procedures.values())[-8:]],
                "global_workspace_recent": self.workspace.broadcast_history[-5:],
                "surprise_stats": {
                    "threshold": self.surprise_evaluator.surprise_threshold,
                    "surprise_ema": round(self.surprise_evaluator.surprise_ema, 4),
                    "total_surprises": self.surprise_evaluator.total_surprises,
                    "pending_queue": self.learner.surprise_queue[:15],
                },
                "mme_council": {
                    "hedge_weights": {k: round(v, 4) for k, v in self.council.hedge_weights.items()},
                    "thompson_ranking": self.council.thompson_sample_ranking(seed=42),
                },
                "cognitive_society": {
                    "rounds_completed": self.society.rounds_completed,
                    "reputation": self.society.reputation.snapshot(),
                    "replicator_proportions": self.society.replicator.proportions,
                },
                "autonomous_acquisitions": self.learner.acquisition_history[-8:],
                "v20_evidence_ledger": self.ledger.summary(),
            }

    def save_state(self) -> None:
        with self._lock:
            try:
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "saved_at": time.time(),
                    "dao_core": self.dao_core.to_state_dict(),
                    "surprise_queue": self.learner.surprise_queue[:100],
                    "hedge_weights": self.council.hedge_weights,
                    "replicator_proportions": self.society.replicator.proportions,
                    "ledger_action_counts": self.ledger.action_counts,
                }
                tmp = self.state_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                tmp.replace(self.state_path)
            except Exception:
                pass

    def load_state(self) -> None:
        with self._lock:
            if not self.state_path.is_file():
                return
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.dao_core.load_state_dict(data.get("dao_core") or {})
                sq = data.get("surprise_queue")
                if isinstance(sq, list):
                    self.learner.surprise_queue = sq
                hw = data.get("hedge_weights")
                if isinstance(hw, dict):
                    for k, v in hw.items():
                        if k in self.council.hedge_weights:
                            self.council.hedge_weights[k] = float(v)
                rp = data.get("replicator_proportions")
                if isinstance(rp, dict):
                    for k, v in rp.items():
                        if k in self.society.replicator.proportions:
                            self.society.replicator.proportions[k] = float(v)
                ac = data.get("ledger_action_counts")
                if isinstance(ac, dict):
                    for k, v in ac.items():
                        self.ledger.action_counts[k] = int(v)
            except Exception:
                pass


_GLOBAL_AGI_ENGINE: Optional[DaxyAgiEngine] = None
_GLOBAL_LOCK = threading.Lock()


def get_agi_engine(brain: Optional[Any] = None) -> DaxyAgiEngine:
    global _GLOBAL_AGI_ENGINE
    with _GLOBAL_LOCK:
        if _GLOBAL_AGI_ENGINE is None:
            _GLOBAL_AGI_ENGINE = DaxyAgiEngine(brain=brain)
        elif brain is not None and _GLOBAL_AGI_ENGINE.brain is None:
            _GLOBAL_AGI_ENGINE.attach_brain(brain)
        return _GLOBAL_AGI_ENGINE
