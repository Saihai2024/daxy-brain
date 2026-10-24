# -*- coding: utf-8 -*-
"""
大兮 AGI (DAXY AGI) 核心认知架构包 (Non-Transformer DAO Cognitive Architecture)
================================================================================

包含六大核心子系统:
  1. core.py              - 非 Transformer 阴阳双态 DAO 认知核心、良心稳态稀疏原型路由 (A20)、三因子局部可塑性与睡眠巩固
  2. mechanism_ir.py      - 一次经验机制获取 (One-Shot Mechanism Acquisition)、ProcedureIR 状态机、跨领域结构同构迁移与反例边界收紧
  3. multimodal_graph.py  - 万物皆可图多模态类型属性图 (Typed Property Graph)、确定性哈希/增量差分、语义拓展合同与最小修复 DAG
  4. workspace_metacog.py - 全局认知工作区 (Global Workspace)、预测加工惊奇触发 (Predictive Surprise)、元认知 EVC/Lyapunov 守卫、多模型校准合议 (MME)
  5. society.py           - 多智能体认知社会: 合同网协议 (ContractNet)、匿名盲审 (Blind Peer Review)、夏普利信用分配 (Shapley)、Beta 声誉与复制子动力学演化
  6. autonomous_learner.py- 惊奇驱动自主多源采集、跨源互证投票、认知免疫反污染、知识图谱自动抽取与 V20 八大行动证据闭环
"""

from .core import DaoConfig, DaoCognitiveCore
from .mechanism_ir import (
    MechanismSignature,
    ProcedureStep,
    ProcedureIR,
    MechanismExtractor,
    StructuralTransferEngine,
    CounterexampleRefiner,
)
from .multimodal_graph import (
    GraphNode,
    GraphEdge,
    GraphDelta,
    TypedPropertyGraph,
    SemanticContract,
    MinimalRepairDAG,
)
from .workspace_metacog import (
    WorkspaceCoalition,
    GlobalCognitiveWorkspace,
    PredictiveCodingSurprise,
    MetacognitiveGuard,
    MultiModelCalibratedCouncil,
)
from .society import (
    ContractNetProtocol,
    BlindPeerReview,
    ShapleyCredit,
    BetaReputation,
    ReplicatorPopulation,
    CognitiveSociety,
)
from .autonomous_learner import (
    V20EvidenceLedger,
    AutonomousLearningLoop,
)

__all__ = [
    "DaoConfig",
    "DaoCognitiveCore",
    "MechanismSignature",
    "ProcedureStep",
    "ProcedureIR",
    "MechanismExtractor",
    "StructuralTransferEngine",
    "CounterexampleRefiner",
    "GraphNode",
    "GraphEdge",
    "GraphDelta",
    "TypedPropertyGraph",
    "SemanticContract",
    "MinimalRepairDAG",
    "WorkspaceCoalition",
    "GlobalCognitiveWorkspace",
    "PredictiveCodingSurprise",
    "MetacognitiveGuard",
    "MultiModelCalibratedCouncil",
    "ContractNetProtocol",
    "BlindPeerReview",
    "ShapleyCredit",
    "BetaReputation",
    "ReplicatorPopulation",
    "CognitiveSociety",
    "V20EvidenceLedger",
    "AutonomousLearningLoop",
]
