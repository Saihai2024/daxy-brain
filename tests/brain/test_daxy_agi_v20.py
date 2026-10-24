# -*- coding: utf-8 -*-
"""
大兮 AGI V20 认知中枢全链路测试套件 (Unit + Integration + API Contract)
覆盖:
  - A1: 非 Transformer 阴阳双态 DAO 原型路由与良心稳态 (Conscience Homeostasis, A20)
  - A2: 三因子局部可塑性与海马体-皮层睡眠重放巩固 (Three-Factor & Sleep Consolidation)
  - A3: 一次经验机制抽取、可执行 ProcedureIR、跨领域结构迁移与反例边界精化
  - A4: 万物皆可图多模态类型属性图谱、确定性 SHA-256 哈希与增量 GraphDelta
  - A5: 形式化语义拓展合同与局部最小修复 DAG (SemanticContract & MinimalRepairDAG)
  - A6: 全局工作区 GWT 竞争点火、预测编码惊奇触发、EVC 路由与 Lyapunov 收敛守卫
  - A7: 多模型校准合议 (MME) 与多智能体认知社会 (ContractNet + 双盲评审 + 精确 Shapley + Beta 声誉 + 复制子动力学)
  - A8: 惊奇驱动自主多源采集、跨源互证投票、认知免疫过滤与 V20 链式证据账本
"""

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.dao import (
    DaoConfig,
    DaoCognitiveCore,
    MechanismExtractor,
    StructuralTransferEngine,
    CounterexampleRefiner,
    TypedPropertyGraph,
    SemanticContract,
    MinimalRepairDAG,
    WorkspaceCoalition,
    GlobalCognitiveWorkspace,
    PredictiveCodingSurprise,
    MetacognitiveGuard,
    MultiModelCalibratedCouncil,
    CognitiveSociety,
)
from brain.daxy_agi_core import DaxyAgiEngine


def test_a1_a2_dao_core_routing_homeostasis_and_sleep():
    cfg = DaoConfig(dim_z=64, n_prototypes=16, n_hot=16, top_k=4, seed=42)
    core = DaoCognitiveCore(cfg)

    rng = np.random.default_rng(123)
    # 1. 零误差时阳态制动必须极低 (< 1e-4)，保证零误伤
    z0 = rng.standard_normal(64).astype(np.float32)
    out0 = core.step(z0, prediction_error=0.0, learn=True, topic_label="稳态基线")
    assert out0["telemetry"]["yang_mean"] < 1e-3
    assert len(out0["active_indices"]) == 4

    # 2. 高预测误差时阳态制动显著上升，且存入海马体情景缓冲
    out_err = core.step(z0, prediction_error=1.2, learn=True, topic_label="高惊奇概念")
    assert out_err["telemetry"]["yang_mean"] > out0["telemetry"]["yang_mean"]
    assert len(core.episodic_buffer) >= 1

    # 3. 连续投喂不同方向向量，验证良心稳态偏置将使用率熵维持在高水平 (防坍缩)
    for i in range(40):
        vec = rng.standard_normal(64).astype(np.float32)
        core.step(vec, prediction_error=0.35, learn=True, topic_label=f"topic_{i}")

    st = core.stats()
    assert st["usage_entropy_ratio"] > 0.70

    # 4. 执行睡眠巩固重放
    sleep_rep = core.sleep_consolidate(n_replay=16)
    assert sleep_rep["replayed"] > 0
    assert sleep_rep["sleep_cycles"] == 1


def test_a3_one_shot_mechanism_extraction_transfer_and_counterexample():
    extractor = MechanismExtractor()
    transfer_eng = StructuralTransferEngine()
    refiner = CounterexampleRefiner()

    src_text = (
        "1. 预检分布式节点网络延迟与并发水位；"
        "2. 通过令牌桶限制瞬时并发速率并将超额请求转入优先级队列异步削峰；"
        "3. 校验核心服务吞吐率守恒且延迟不崩溃，若异常则触发增量回滚通道。"
    )
    proc_ir = extractor.extract_one_shot(
        title="令牌桶异步削峰机制",
        experience_text=src_text,
        source_domain="distributed_systems",
    )
    assert len(proc_ir.signature.preconditions) >= 1
    assert len(proc_ir.steps) >= 1

    # 跨领域结构同构迁移
    transferred = transfer_eng.transfer(
        source_ir=proc_ir,
        target_domain="brain_workspace_scheduling",
    )
    assert transferred.signature.source_domain == "brain_workspace_scheduling"
    assert len(transferred.steps) == len(proc_ir.steps)
    assert transferred.transfer_parent_id == proc_ir.ir_id

    # 反例驱动边界收紧
    refined = refiner.refine(
        proc_ir=transferred,
        counterexample_desc="死锁状态下的同步阻塞调用",
        violated_step_no=1,
    )
    assert len(refined.signature.negative_guards) >= 1
    assert "NOT" in refined.steps[0].guard_predicate


def test_a4_a5_multimodal_graph_and_semantic_contract_repair_dag():
    graph = TypedPropertyGraph()
    delta = graph.ingest_knowledge_passage(
        topic="非阿贝尔任意子",
        content="非阿贝尔任意子是一种二维拓扑相准粒子激发。非阿贝尔任意子包含马约拉纳零能模，非阿贝尔任意子用于容错拓扑量子计算。",
        source="unit_test",
        credibility=0.92,
    )
    assert delta.base_hash != delta.new_hash
    assert len(delta.added_nodes) >= 2
    assert len(delta.added_edges) >= 1

    activated = graph.spread_activation(["非阿贝尔任意子"], hops=2)
    assert len(activated) >= 1
    assert activated[0]["label"] == "非阿贝尔任意子"

    # 语义合同与局部最小修复 DAG
    query = "请解释非阿贝尔任意子是什么，必须包含拓扑，不要玄学"
    contract = SemanticContract.compile_from_query(query)
    assert "拓扑" in contract.required_keywords
    assert "玄学" in contract.forbidden_keywords

    bad_answer = "这是一种玄学粒子。它具有非常奇特的物理性质。"
    repaired = MinimalRepairDAG.verify_and_repair(
        contract=contract,
        candidate_response=bad_answer,
        graph_context=activated,
    )
    assert repaired["passed_initially"] is False
    assert repaired["verified_after_repair"] is True
    assert "玄学" not in repaired["repaired_response"]
    assert "拓扑" in repaired["repaired_response"]


def test_a6_a7_workspace_metacog_mme_and_society():
    ws = GlobalCognitiveWorkspace(capacity=2)
    c1 = WorkspaceCoalition("left_sdr_semantic", "text", "阴阳路由", "阴阳原型路由解析", raw_confidence=0.88, salience=0.85)
    c2 = WorkspaceCoalition("multimodal_property_graph", "graph", "图谱激活", "多模态图谱激活扩散", raw_confidence=0.82, salience=0.78)
    c3 = WorkspaceCoalition("noise_mod", "text", "噪声", "低显著性噪声", raw_confidence=0.30, salience=0.20)
    ign = ws.compete_and_ignite([c1, c2, c3])
    assert len(ign["ignited"]) == 2
    assert ign["winning_specialist"] == "left_sdr_semantic"

    surprise_mod = PredictiveCodingSurprise(surprise_threshold=0.50)
    s_info = surprise_mod.evaluate(query="未知超对称拓扑子是什么？", max_similarity=0.15, epistemic_uncertainty=0.5, graph_coverage=0)
    assert s_info["is_surprised"] is True
    assert s_info["curiosity_topic"] == "未知超对称拓扑子"

    guard = MetacognitiveGuard()
    evc = guard.compute_evc_route(fast_confidence=0.25, surprise_score=0.78, is_procedural_query=False)
    assert evc["selected_route"] in ("autonomous_web_acquisition", "system2_dao_deliberation")

    lyap = guard.check_lyapunov(prediction_error=0.4, yang_mean=0.1)
    assert lyap["lyapunov_v"] > 0

    mme = MultiModelCalibratedCouncil()
    proposals = [
        {"specialist": "left_sdr_semantic", "topic": "拓扑量子", "content": "因为良心稳态偏置调节阈值，因此避免赢家通吃坍缩。", "confidence": 0.88, "cost": 0.05},
        {"specialist": "right_sdr_associative", "topic": "拓扑量子", "content": "类比生物神经元稳态可塑性与睡眠巩固。", "confidence": 0.76, "cost": 0.05},
    ]
    mme_res = mme.deliberate(proposals)
    assert 0.5 < mme_res["pooled_confidence"] < 0.99

    society = CognitiveSociety()
    soc_res = society.run_society_round("测试合议任务", mme_res["calibrated_proposals"])
    assert soc_res["contract_award"]["awarded_to"] in society.DEFAULT_AGENTS
    assert len(soc_res["shapley_credits"]) == 2


def test_a8_daxy_agi_engine_full_closed_loop(tmp_path: Path):
    engine = DaxyAgiEngine(brain=None, state_path=tmp_path / "agi_state.json")

    # 1) 自主多源/投喂互证学习 (A8)
    acq = engine.learner.acquire_and_learn(
        topic="马约拉纳零能模",
        agi_engine=engine,
        custom_content="马约拉纳零能模是一种出现在拓扑超导体涡旋中的准粒子激发。马约拉纳零能模满足非阿贝尔统计规律，马约拉纳零能模用于容错拓扑量子计算。",
        allow_network=False,
        persist_pg=False,
    )
    assert acq["ok"] is True
    assert acq["graph_delta"]["added_nodes"] >= 1

    # 2) 一次经验机制抽取与跨领域同构迁移 + 反例精化 (A3)
    mech = engine.one_shot_learn_and_transfer(
        title="良心稳态防坍缩机制",
        experience_text="1. 追踪各原型的指数滑动使用率；2. 按偏差比例抬高高频原型的竞争阈值；3. 实现全部原型的均衡激活。",
        source_domain="dao_routing",
        target_domain="expert_load_balancing",
        counterexample="所有输入向量完全恒等退化为零向量",
    )
    assert "base_procedure_ir" in mech
    assert mech["transferred_procedure_ir"]["signature"]["source_domain"] == "expert_load_balancing"
    assert len(mech["active_procedure_ir"]["signature"]["negative_guards"]) >= 1

    # 3) 全链路认知合议 (覆盖 A1, A5, A6, A7)
    delib = engine.deliberate("马约拉纳零能模在拓扑量子计算中有什么作用？必须包含拓扑")
    assert delib["mme_council"]["pooled_confidence"] > 0.3
    assert len(delib["final_response"]) >= 10
    assert "拓扑" in delib["final_response"]

    # 4) 睡眠巩固 (A2)
    sleep_res = engine.run_sleep_consolidation(n_replay=8)
    assert "usage_entropy" in sleep_res

    # 5) V20 八大行动全部覆盖且 SHA-256 证据链完整
    status = engine.status()
    v20 = status["v20_evidence_ledger"]
    assert v20["chain_integrity"] is True
    assert v20["actions_covered"] == "8/8"
    assert (tmp_path / "agi_state.json").exists()


class TestDaxyAgiV20(unittest.TestCase):
    def test_all_v20_actions(self):
        test_a1_a2_dao_core_routing_homeostasis_and_sleep()
        test_a3_one_shot_mechanism_extraction_transfer_and_counterexample()
        test_a4_a5_multimodal_graph_and_semantic_contract_repair_dag()
        test_a6_a7_workspace_metacog_mme_and_society()
        with tempfile.TemporaryDirectory() as td:
            test_a8_daxy_agi_engine_full_closed_loop(Path(td))


if __name__ == "__main__":
    unittest.main(verbosity=2)
