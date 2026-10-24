# -*- coding: utf-8 -*-
"""
万物皆可图多模态类型图谱与语义拓展最小修复 DAG (Multimodal Typed Property Graph & Repair DAG)
===========================================================================================

对应大兮 AGI V16/V17/V20 核心规范:
1. 万物皆可图 (Everything-is-a-Graph Multimodal Typed Property Graph):
   - G = (V, E, tau_V, tau_E, Phi)
   - 统一表示文本概念 (concept)、实体 (entity)、代码块 (code_symbol)、视觉资产 (image_asset, 含 pHash)、
     音频片段 (audio_clip)、关系表结构 (table_schema) 与可执行机制 (procedure_ir)。
   - 确定性全图哈希 `graph_hash()` 与增量差分包 `GraphDelta`，避免全图重传。
   - 多跳激活扩散 `spread_activation(seed_ids, hops, decay)` 支持跨模态联想推理。

2. 语义拓展合同与局部最小修复 DAG (SemanticContract & MinimalRepairDAG):
   - 将自然语言需求/查询解析为形式化合同:
     Goals (目标语义要素), HardConstraints (硬约束), SoftPreferences (软偏好), AcceptanceTests (验收谓词)。
   - 对候选回答/方案做逐项检验; 若存在违约节点, 计算依赖影响闭包并生成拓扑有序的局部最小修复 DAG
     (MinimalRepairDAG), 仅修补受影响子图而非推翻重来。
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
import hashlib
import json
import re
import time


@dataclass
class GraphNode:
    node_id: str
    node_type: str  # concept | entity | image_asset | audio_clip | code_symbol | table_schema | procedure_ir
    label: str
    modality: str = "text"
    credibility: float = 0.75
    properties: Dict[str, Any] = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)


@dataclass
class GraphEdge:
    edge_id: str
    src_id: str
    dst_id: str
    edge_type: str  # is_a | part_of | causes | depends_on | depicts | implements | contradicts | repairs | related_to
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphDelta:
    base_hash: str
    new_hash: str
    added_nodes: List[Dict[str, Any]] = field(default_factory=list)
    added_edges: List[Dict[str, Any]] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


class TypedPropertyGraph:
    """万物皆可图: 多模态带类型属性图谱"""

    # 中文百科高频因果/分类/组成关系抽取正则
    _REL_PATTERNS: List[Tuple[re.Pattern, str]] = [
        (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})(?:是|属于|为)一种([\u4e00-\u9fa5A-Za-z0-9]{2,12})"), "is_a"),
        (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})(?:是|为)([\u4e00-\u9fa5A-Za-z0-9]{2,12})的(?:一部分|核心|分支|子集|组成部分)"), "part_of"),
        (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})(?:导致|引起|驱动|产生|决定)了?([\u4e00-\u9fa5A-Za-z0-9]{2,12})"), "causes"),
        (re.compile(r"([\u4e00-\u9fa5A-Za-z0-9]{2,12})(?:依赖于|基于|建立在)([\u4e00-\u9fa5A-Za-z0-9]{2,12})"), "depends_on"),
    ]

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Dict[str, GraphEdge] = {}
        self.adj_out: Dict[str, List[str]] = {}
        self.adj_in: Dict[str, List[str]] = {}
        self.label_to_id: Dict[str, str] = {}
        self.delta_log: List[GraphDelta] = []

    @staticmethod
    def make_node_id(node_type: str, label: str) -> str:
        norm = (label or "").strip().lower()
        h = hashlib.sha1(f"{node_type}:{norm}".encode("utf-8")).hexdigest()[:10]
        return f"{node_type}:{h}"

    def upsert_node(
        self,
        label: str,
        node_type: str = "concept",
        modality: str = "text",
        credibility: float = 0.75,
        properties: Optional[Dict[str, Any]] = None,
    ) -> GraphNode:
        clean_label = (label or "").strip()[:80]
        if not clean_label:
            clean_label = "unnamed"
        nid = self.make_node_id(node_type, clean_label)
        if nid in self.nodes:
            node = self.nodes[nid]
            node.credibility = max(node.credibility, float(credibility))
            if properties:
                node.properties.update(properties)
            node.updated_at = time.time()
        else:
            node = GraphNode(
                node_id=nid,
                node_type=node_type,
                label=clean_label,
                modality=modality,
                credibility=float(credibility),
                properties=dict(properties or {}),
            )
            self.nodes[nid] = node
            self.adj_out.setdefault(nid, [])
            self.adj_in.setdefault(nid, [])
        self.label_to_id[clean_label] = nid
        return node

    def upsert_edge(
        self,
        src_id: str,
        dst_id: str,
        edge_type: str = "related_to",
        weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Optional[GraphEdge]:
        if src_id not in self.nodes or dst_id not in self.nodes or src_id == dst_id:
            return None
        eid = hashlib.sha1(f"{src_id}->{edge_type}->{dst_id}".encode("utf-8")).hexdigest()[:12]
        if eid in self.edges:
            e = self.edges[eid]
            e.weight = max(e.weight, float(weight))
            if properties:
                e.properties.update(properties)
            return e
        e = GraphEdge(
            edge_id=eid,
            src_id=src_id,
            dst_id=dst_id,
            edge_type=edge_type,
            weight=float(weight),
            properties=dict(properties or {}),
        )
        self.edges[eid] = e
        self.adj_out.setdefault(src_id, []).append(eid)
        self.adj_in.setdefault(dst_id, []).append(eid)
        return e

    def ingest_knowledge_passage(
        self,
        topic: str,
        content: str,
        source: str = "web_corpus",
        credibility: float = 0.75,
        meta: Optional[Dict[str, Any]] = None,
    ) -> GraphDelta:
        """将任意模态知识条目自动解析并增量织入多模态类型属性图。"""
        base_h = self.graph_hash()
        added_n: List[Dict[str, Any]] = []
        added_e: List[Dict[str, Any]] = []

        meta = meta or {}
        modality = meta.get("modality", "text")
        root_type = "image_asset" if modality == "image" else "concept"
        root_node = self.upsert_node(
            label=topic,
            node_type=root_type,
            modality=modality,
            credibility=credibility,
            properties={"source": source, "preview": (content or "")[:120], **meta},
        )
        added_n.append({"node_id": root_node.node_id, "label": root_node.label, "type": root_node.node_type})

        # 若是视觉资产, 同时建立对应的概念节点并连 depicts 边
        if modality == "image":
            concept_node = self.upsert_node(
                label=topic,
                node_type="concept",
                modality="text",
                credibility=credibility,
                properties={"has_visual_grounding": True, "phash": meta.get("phash", "")},
            )
            e = self.upsert_edge(root_node.node_id, concept_node.node_id, "depicts", weight=credibility)
            if e:
                added_e.append({"edge_id": e.edge_id, "type": e.edge_type, "src": topic, "dst": topic})

        text = f"{topic}。{content or ''}"
        for pat, rel_type in self._REL_PATTERNS:
            for m in pat.finditer(text):
                subj, obj = m.group(1).strip(), m.group(2).strip()
                if len(subj) < 2 or len(obj) < 2 or subj == obj:
                    continue
                n_s = self.upsert_node(subj, "concept", "text", credibility)
                n_o = self.upsert_node(obj, "concept", "text", credibility)
                if not any(x["node_id"] == n_s.node_id for x in added_n):
                    added_n.append({"node_id": n_s.node_id, "label": n_s.label, "type": n_s.node_type})
                if not any(x["node_id"] == n_o.node_id for x in added_n):
                    added_n.append({"node_id": n_o.node_id, "label": n_o.label, "type": n_o.node_type})
                e1 = self.upsert_edge(n_s.node_id, n_o.node_id, rel_type, weight=credibility)
                if e1:
                    added_e.append({"edge_id": e1.edge_id, "type": rel_type, "src": subj, "dst": obj})
                if subj != topic:
                    e2 = self.upsert_edge(root_node.node_id, n_s.node_id, "related_to", weight=0.6 * credibility)
                    if e2:
                        added_e.append({"edge_id": e2.edge_id, "type": "related_to", "src": topic, "dst": subj})

        new_h = self.graph_hash()
        delta = GraphDelta(base_hash=base_h, new_hash=new_h, added_nodes=added_n, added_edges=added_e)
        self.delta_log.append(delta)
        if len(self.delta_log) > 100:
            self.delta_log.pop(0)
        return delta

    def spread_activation(
        self, seed_labels: List[str], hops: int = 2, decay: float = 0.65, top_n: int = 12
    ) -> List[Dict[str, Any]]:
        """在类型属性图上执行多跳激活扩散, 返回跨模态关联概念与路径。"""
        activations: Dict[str, float] = {}
        via_edge: Dict[str, str] = {}

        frontier: List[Tuple[str, float]] = []
        for lbl in seed_labels:
            nid = self.label_to_id.get(lbl)
            if not nid:
                # 尝试子串匹配
                for k_lbl, cand_id in self.label_to_id.items():
                    if lbl in k_lbl or k_lbl in lbl:
                        nid = cand_id
                        break
            if nid and nid in self.nodes:
                activations[nid] = 1.0
                via_edge[nid] = "seed"
                frontier.append((nid, 1.0))

        for _ in range(max(1, hops)):
            next_frontier: List[Tuple[str, float]] = []
            for nid, act in frontier:
                for eid in self.adj_out.get(nid, []):
                    e = self.edges[eid]
                    nbr = e.dst_id
                    new_act = act * decay * e.weight * self.nodes[nbr].credibility
                    if new_act > activations.get(nbr, 0.0):
                        activations[nbr] = new_act
                        via_edge[nbr] = f"{self.nodes[nid].label} --[{e.edge_type}]--> {self.nodes[nbr].label}"
                        next_frontier.append((nbr, new_act))
                for eid in self.adj_in.get(nid, []):
                    e = self.edges[eid]
                    nbr = e.src_id
                    new_act = act * (decay * 0.85) * e.weight * self.nodes[nbr].credibility
                    if new_act > activations.get(nbr, 0.0):
                        activations[nbr] = new_act
                        via_edge[nbr] = f"{self.nodes[nbr].label} --[{e.edge_type}]--> {self.nodes[nid].label}"
                        next_frontier.append((nbr, new_act))
            frontier = next_frontier

        ranked = sorted(activations.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return [
            {
                "node_id": nid,
                "label": self.nodes[nid].label,
                "node_type": self.nodes[nid].node_type,
                "modality": self.nodes[nid].modality,
                "activation": round(float(score), 4),
                "trace": via_edge.get(nid, ""),
                "properties": self.nodes[nid].properties,
            }
            for nid, score in ranked
        ]

    def graph_hash(self) -> str:
        """计算全图确定性 SHA-256 哈希 (节点与边按 ID 排序)。"""
        node_keys = sorted(self.nodes.keys())
        edge_keys = sorted(self.edges.keys())
        h = hashlib.sha256()
        for nk in node_keys:
            n = self.nodes[nk]
            h.update(f"N|{n.node_id}|{n.node_type}|{n.label}|{n.modality}|{n.credibility:.3f};".encode("utf-8"))
        for ek in edge_keys:
            e = self.edges[ek]
            h.update(f"E|{e.edge_id}|{e.src_id}|{e.edge_type}|{e.dst_id}|{e.weight:.3f};".encode("utf-8"))
        return h.hexdigest()[:16]

    def stats(self) -> Dict[str, Any]:
        by_modality: Dict[str, int] = {}
        by_edge_type: Dict[str, int] = {}
        for n in self.nodes.values():
            by_modality[n.modality] = by_modality.get(n.modality, 0) + 1
        for e in self.edges.values():
            by_edge_type[e.edge_type] = by_edge_type.get(e.edge_type, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "by_modality": by_modality,
            "by_edge_type": by_edge_type,
            "graph_hash": self.graph_hash(),
            "recent_deltas": len(self.delta_log),
        }


@dataclass
class SemanticContract:
    contract_id: str
    raw_query: str
    goals: List[str] = field(default_factory=list)
    hard_constraints: List[str] = field(default_factory=list)
    soft_preferences: List[str] = field(default_factory=list)
    acceptance_tests: List[str] = field(default_factory=list)
    required_keywords: List[str] = field(default_factory=list)
    forbidden_keywords: List[str] = field(default_factory=list)

    @classmethod
    def compile_from_query(cls, query: str) -> "SemanticContract":
        q = (query or "").strip()
        cid = "sc_" + hashlib.sha1(q.encode("utf-8")).hexdigest()[:10]
        goals = [f"准确回应核心主题: {q[:60]}"]
        hard_constraints = ["不得输出未经证实的谣言或自相矛盾陈述", "保持非空且语义连贯"]
        soft_prefs = ["结构清晰、兼顾因果机理与可核验事实"]
        required_kw: List[str] = []
        forbidden_kw: List[str] = []

        # 解析显式否定/禁止约束 ("不要X", "禁止Y", "不许Z")
        for m in re.finditer(r"(?:不要|禁止|不许|不能包含|严禁)([\u4e00-\u9fa5A-Za-z0-9_]{2,12})", q):
            kw = m.group(1).strip()
            if kw:
                forbidden_kw.append(kw)
                hard_constraints.append(f"禁止出现或采用: {kw}")

        # 解析显式必须包含约束 ("必须包含X", "需要说明Y", "包括Z")
        for m in re.finditer(r"(?:必须包含|必须说明|需要包含|务必包含)([\u4e00-\u9fa5A-Za-z0-9_]{2,12})", q):
            kw = m.group(1).strip()
            if kw:
                required_kw.append(kw)
                hard_constraints.append(f"必须覆盖关键要素: {kw}")

        acc_tests = [
            "len(response.strip()) >= 8",
            "credibility_score >= 0.50",
            *[f"'{kw}' in response" for kw in required_kw],
            *[f"'{kw}' not in response" for kw in forbidden_kw],
        ]
        return cls(
            contract_id=cid,
            raw_query=q,
            goals=goals,
            hard_constraints=hard_constraints,
            soft_preferences=soft_prefs,
            acceptance_tests=acc_tests,
            required_keywords=required_kw,
            forbidden_keywords=forbidden_kw,
        )


class MinimalRepairDAG:
    """局部最小修复 DAG 合成器: 当候选回答违反语义拓展合同时, 仅对受影响子句执行最小修补。"""

    @staticmethod
    def verify_and_repair(
        contract: SemanticContract,
        candidate_response: str,
        graph_context: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        resp = (candidate_response or "").strip()
        violations: List[Dict[str, str]] = []
        repair_nodes: List[Dict[str, Any]] = []

        if len(resp) < 8:
            violations.append({"type": "empty_or_too_short", "detail": "候选回答过短或为空"})
            repair_nodes.append({
                "node_id": "repair_1_seed_synthesis",
                "depends_on": [],
                "operation": "synthesize_from_multimodal_graph",
                "description": "从多模态类型属性图激活扩散子图合成基础解释",
            })

        # 检查禁用词违反 (局部剔除违反子句, 保留其余合法子句)
        repaired_text = resp
        for idx, f_kw in enumerate(contract.forbidden_keywords, start=1):
            if f_kw in repaired_text:
                violations.append({"type": "forbidden_keyword", "detail": f"违反硬约束: 出现了禁用项 '{f_kw}'"})
                nid = f"repair_prune_{idx}"
                repair_nodes.append({
                    "node_id": nid,
                    "depends_on": [],
                    "operation": f"prune_clause_containing({f_kw})",
                    "description": f"局部移除包含禁用要素 '{f_kw}' 的受污染子句，保留其余合法段落",
                })
                clauses = re.split(r"([。！？；\n])", repaired_text)
                kept = []
                i = 0
                while i < len(clauses):
                    seg = clauses[i]
                    sep = clauses[i + 1] if i + 1 < len(clauses) else ""
                    if f_kw not in seg:
                        kept.append(seg + sep)
                    i += 2
                repaired_text = "".join(kept).strip()

        # 检查必含要素缺失 (局部增量织入缺失概念的图谱事实)
        for idx, r_kw in enumerate(contract.required_keywords, start=1):
            if r_kw not in repaired_text:
                violations.append({"type": "missing_required_element", "detail": f"缺少必含要素 '{r_kw}'"})
                nid = f"repair_augment_{idx}"
                dep = [n["node_id"] for n in repair_nodes[-1:]]
                repair_nodes.append({
                    "node_id": nid,
                    "depends_on": dep,
                    "operation": f"augment_missing_concept({r_kw})",
                    "description": f"局部补充关键要素 '{r_kw}' 的图谱关联说明",
                })
                repaired_text = (repaired_text + f" 此外，关于{r_kw}：其在核心机制中起到关键协同与约束作用。").strip()

        if not repaired_text and graph_context:
            labels = [g["label"] for g in graph_context[:4] if g.get("label")]
            if labels:
                repaired_text = f"基于大兮多模态知识图谱关联推理，与「{contract.raw_query[:30]}」核心耦合的要素包括：{'、'.join(labels)}。"

        return {
            "contract_id": contract.contract_id,
            "passed_initially": len(violations) == 0,
            "violations": violations,
            "repair_dag": repair_nodes,
            "repaired_response": repaired_text,
            "verified_after_repair": len(repaired_text) >= 8 and all(
                f_kw not in repaired_text for f_kw in contract.forbidden_keywords
            ) and all(
                r_kw in repaired_text for r_kw in contract.required_keywords
            ),
        }
