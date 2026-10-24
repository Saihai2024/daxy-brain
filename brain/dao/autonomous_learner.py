# -*- coding: utf-8 -*-
"""
惊奇驱动自主多源采集学习闭环与 V20 八大行动证据账本
(Autonomous Multi-Source Learning Loop & V20 8-Action Evidence Ledger)
======================================================================

对应大兮 AGI V14-V20 自主持续学习与治理规范:
1. V20 八大行动证据闭环 (V20EvidenceLedger):
   - A1: non_transformer_dao_routing       (非 Transformer 阴阳双态原型路由与良心稳态)
   - A2: three_factor_sleep_consolidation  (三因子局部可塑性与双系统睡眠巩固)
   - A3: one_shot_mechanism_transfer       (一次经验机制抽取与跨领域结构迁移)
   - A4: multimodal_typed_property_graph   (万物皆可图多模态类型图谱与增量差分)
   - A5: semantic_contract_repair_dag      (语义拓展合同与局部最小修复 DAG)
   - A6: global_workspace_evc_metacog      (全局工作区点火、预测惊奇与 EVC/Lyapunov 守卫)
   - A7: mme_council_cognitive_society     (多模型校准合议与多智能体社会盲审/Shapley/声誉)
   - A8: autonomous_corroborated_learning  (惊奇驱动多源互证采集与认知免疫入脑)
   每条证据记录绑定 prev_hash -> pack_hash 不可篡改 SHA-256 哈希链，并支持同步落库 PG。

2. 惊奇驱动自主学习闭环 (AutonomousLearningLoop):
   - 当对话或推理遇到高惊奇度 (S_t >= tau_surprise) 时，自动提取核心未知主题推入优先求知队列。
   - 调用多源采集器 (`skills.multi_source_crawler.fetch_topic(strategy='multi')`) 从维基百科、arXiv、
     百度百科、360百科、搜狗百科、萌娘百科、Bing、DuckDuckGo 并发/顺序拉取多源候选。
   - 跨源互证投票 (Cross-Source Corroboration Voting):
     计算多源文本的核心实义词重叠度；若 >=2 个独立来源互证一致，可信度获得协同加成 (+0.08~+0.15)；
     再经认知免疫系统 (`skills.immune_system.CognitiveImmuneSystem`) 与 `brain.credibility.score` 双重过滤。
   - 自动将高可信知识同步织入:
     (a) PostgreSQL `web_corpus` / `sentence_corpus` / `thinking_gaps`
     (b) `CognitiveBrain` SDR 双脑记忆库
     (c) `DaoCognitiveCore` 阴阳双态原型库
     (d) `TypedPropertyGraph` 多模态知识图谱
     (e) `ProcedureIR` 机制库
"""

from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json
import time


class V20EvidenceLedger:
    """V20 八大行动 SHA-256 链式证据账本"""

    V20_ACTIONS = (
        "A1_non_transformer_dao_routing",
        "A2_three_factor_sleep_consolidation",
        "A3_one_shot_mechanism_transfer",
        "A4_multimodal_typed_property_graph",
        "A5_semantic_contract_repair_dag",
        "A6_global_workspace_evc_metacog",
        "A7_mme_council_cognitive_society",
        "A8_autonomous_corroborated_learning",
    )

    def __init__(self):
        self.chain: List[Dict[str, Any]] = []
        self.action_counts: Dict[str, int] = {a: 0 for a in self.V20_ACTIONS}

    @property
    def head_hash(self) -> str:
        return self.chain[-1]["pack_hash"] if self.chain else ("0" * 16)

    def record(self, action_id: str, subject: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        prev_h = self.head_hash
        ts = round(time.time(), 3)
        canonical = json.dumps(
            {"prev": prev_h, "action": action_id, "subject": subject[:80], "metrics": metrics},
            sort_keys=True,
            ensure_ascii=False,
        )
        pack_h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        entry = {
            "seq": len(self.chain) + 1,
            "action_id": action_id,
            "subject": subject[:80],
            "metrics": metrics,
            "prev_hash": prev_h,
            "pack_hash": pack_h,
            "ts": ts,
        }
        self.chain.append(entry)
        if len(self.chain) > 200:
            self.chain.pop(0)
        if action_id in self.action_counts:
            self.action_counts[action_id] += 1
        else:
            self.action_counts[action_id] = 1
        return entry

    def verify_integrity(self) -> bool:
        for i in range(1, len(self.chain)):
            if self.chain[i]["prev_hash"] != self.chain[i - 1]["pack_hash"]:
                return False
        return True

    def summary(self) -> Dict[str, Any]:
        covered = sum(1 for v in self.action_counts.values() if v > 0)
        return {
            "total_records": len(self.chain),
            "head_hash": self.head_hash,
            "chain_integrity": self.verify_integrity(),
            "actions_covered": f"{covered}/{len(self.V20_ACTIONS)}",
            "action_counts": dict(self.action_counts),
            "recent_packs": self.chain[-8:],
        }


class AutonomousLearningLoop:
    """惊奇驱动自主多源采集、跨源互证投票与认知免疫闭环"""

    def __init__(self, evidence_ledger: V20EvidenceLedger):
        self.ledger = evidence_ledger
        self.surprise_queue: List[Dict[str, Any]] = []
        self.acquisition_history: List[Dict[str, Any]] = []

    def enqueue_surprise_gap(
        self,
        topic: str,
        raw_query: str,
        surprise_score: float,
        persist_pg: bool = True,
    ) -> Optional[Dict[str, Any]]:
        clean_t = (topic or "").strip()[:80]
        if len(clean_t) < 2:
            return None

        # 避免重复入队
        for item in self.surprise_queue:
            if item["topic"] == clean_t:
                item["hits"] += 1
                item["surprise_score"] = max(item["surprise_score"], round(float(surprise_score), 4))
                item["last_seen"] = time.time()
                return item

        entry = {
            "topic": clean_t,
            "query": (raw_query or clean_t)[:200],
            "surprise_score": round(float(surprise_score), 4),
            "hits": 1,
            "status": "pending",
            "created_at": time.time(),
            "last_seen": time.time(),
        }
        self.surprise_queue.append(entry)
        self.surprise_queue.sort(key=lambda x: (x["status"] == "pending", x["surprise_score"], x["hits"]), reverse=True)
        if len(self.surprise_queue) > 200:
            self.surprise_queue = self.surprise_queue[:200]

        if persist_pg:
            try:
                from utils.db_config import get_pg_conn
                conn = get_pg_conn()
                try:
                    with conn, conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO thinking_gaps (query, hits, best_rel, status) "
                            "VALUES (%s, 1, %s, 'pending') "
                            "ON CONFLICT (query) DO UPDATE SET "
                            "  hits = thinking_gaps.hits + 1, "
                            "  last_seen = now()",
                            (entry["query"], max(0.0, 1.0 - float(surprise_score))),
                        )
                finally:
                    conn.close()
            except Exception:
                pass
        return entry

    @staticmethod
    def _bigram_set(text: str) -> set:
        s = "".join(ch for ch in (text or "") if "\u4e00" <= ch <= "\u9fa5" or ch.isalnum())
        return {s[i:i + 2] for i in range(max(0, len(s) - 1))}

    def corroborate_sources(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """跨源互证投票 (Cross-Source Corroboration Voting):
        对多源抓取结果计算两两语义重叠度，赋予多源印证加成并选出最佳融合文本。
        """
        if not candidates:
            return {"corroborated": False, "source_count": 0, "best": None, "sources": []}

        try:
            from skills.immune_system import CognitiveImmuneSystem
            immune = CognitiveImmuneSystem()
        except Exception:
            immune = None

        bg_sets = [self._bigram_set(c.get("content", "")) for c in candidates]
        n = len(candidates)
        scored = []
        for i, cand in enumerate(candidates):
            txt = cand.get("content", "")
            base_cred = float(cand.get("credibility", 0.65))
            imm_score = immune.evaluate_credibility(txt) if immune else 0.85

            # 与其他独立源的平均重叠互证度
            overlaps = []
            for j in range(n):
                if i == j or not bg_sets[i] or not bg_sets[j]:
                    continue
                inter = len(bg_sets[i] & bg_sets[j])
                denom = max(1, min(len(bg_sets[i]), len(bg_sets[j])))
                overlaps.append(inter / float(denom))
            avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
            corroboration_bonus = min(0.15, 0.06 * max(0, n - 1) + 0.12 * avg_overlap)

            final_cred = min(0.98, 0.55 * base_cred + 0.35 * imm_score + corroboration_bonus)
            utility = final_cred * math.log1p(min(len(txt), 2500))
            scored.append({
                "source": cand.get("source", "unknown"),
                "content": txt,
                "base_credibility": round(base_cred, 4),
                "immune_score": round(imm_score, 4),
                "corroboration_overlap": round(avg_overlap, 4),
                "corroboration_bonus": round(corroboration_bonus, 4),
                "final_credibility": round(final_cred, 4),
                "utility": round(utility, 4),
                "image_url": cand.get("image_url"),
            })

        scored.sort(key=lambda x: x["utility"], reverse=True)
        best = scored[0]
        return {
            "corroborated": n >= 2 and best["corroboration_overlap"] >= 0.05,
            "source_count": n,
            "best": best,
            "sources": [
                {
                    "source": s["source"],
                    "final_credibility": s["final_credibility"],
                    "length": len(s["content"]),
                    "overlap": s["corroboration_overlap"],
                }
                for s in scored
            ],
        }

    def acquire_and_learn(
        self,
        topic: str,
        agi_engine: Any,
        custom_content: Optional[str] = None,
        allow_network: bool = True,
        persist_pg: bool = True,
    ) -> Dict[str, Any]:
        """执行完整的『多源采集 -> 跨源互证 -> 免疫过滤 -> 多模态图谱抽取 -> DAO 原型与双脑巩固 -> V20 证据入账』闭环。"""
        clean_topic = (topic or "").strip()[:80]
        if not clean_topic:
            return {"ok": False, "reason": "empty_topic"}

        candidates: List[Dict[str, Any]] = []
        if custom_content and len(custom_content.strip()) >= 15:
            candidates.append({
                "content": custom_content.strip(),
                "source": "direct_teaching",
                "credibility": 0.88,
                "image_url": None,
            })

        if allow_network and not candidates:
            try:
                from skills.multi_source_crawler import fetch_topic
                fetched = fetch_topic(clean_topic, strategy="multi", allow_browser=False)
                if isinstance(fetched, list):
                    candidates.extend(fetched)
                elif isinstance(fetched, dict):
                    candidates.append(fetched)
            except Exception:
                pass

        if not candidates:
            return {"ok": False, "topic": clean_topic, "reason": "no_sources_returned"}

        corr = self.corroborate_sources(candidates)
        best = corr["best"]
        if not best:
            return {"ok": False, "topic": clean_topic, "reason": "corroboration_empty"}

        content = best["content"]
        final_cred = float(best["final_credibility"])
        src_name = best["source"]

        # 额外通过 brain.credibility 规则核验
        try:
            from brain.credibility import score as rule_cred_score
            rc = rule_cred_score(clean_topic, content, source="web_corpus")
            if rc > 0:
                final_cred = round(0.65 * final_cred + 0.35 * rc, 4)
        except Exception:
            pass

        if final_cred < 0.35:
            return {
                "ok": False,
                "topic": clean_topic,
                "reason": "quarantined_by_immune_system",
                "credibility": final_cred,
            }

        # 1. 织入万物皆可图多模态属性图谱
        graph_delta = agi_engine.graph.ingest_knowledge_passage(
            topic=clean_topic,
            content=content,
            source=src_name,
            credibility=final_cred,
        )

        # 2. 抽取单样本因果机制 ProcedureIR
        proc_ir = agi_engine.extractor.extract_one_shot(
            title=clean_topic,
            experience_text=content,
            source_domain=src_name,
        )
        agi_engine.register_procedure(proc_ir)

        # 3. 编码进入 CognitiveBrain 与 DAO 阴阳双态原型核心
        brain_idx = None
        dao_step_info = {}
        if agi_engine.brain is not None:
            try:
                key_vec = agi_engine.brain.encode(clean_topic)
                dao_res = agi_engine.dao_core.step(
                    key_vec,
                    prediction_error=max(0.0, 1.0 - final_cred),
                    learn=True,
                    topic_label=clean_topic,
                )
                dao_step_info = dao_res["telemetry"]
                brain_idx = agi_engine.brain.add_memory(
                    clean_topic,
                    content,
                    source=f"agi_auto:{src_name}",
                    credibility=final_cred,
                )
            except Exception:
                pass

        # 4. 持久化到 PostgreSQL web_corpus & sentence_corpus
        pg_saved = False
        if persist_pg:
            try:
                from utils.db_config import get_pg_conn
                conn = get_pg_conn()
                try:
                    with conn, conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO web_corpus (topic, content, status, credibility) "
                            "VALUES (%s, %s, 'learned', %s)",
                            (clean_topic, content, final_cred),
                        )
                        rehearse_q = (clean_topic + "是什么")[:255]
                        cur.execute(
                            "INSERT INTO sentence_corpus (topic, content, status, credibility) "
                            "VALUES (%s, %s, 'learned', %s) "
                            "ON CONFLICT (topic) DO UPDATE SET "
                            "  content = EXCLUDED.content, "
                            "  credibility = GREATEST(sentence_corpus.credibility, EXCLUDED.credibility), "
                            "  status = 'learned'",
                            (rehearse_q, content, final_cred),
                        )
                        cur.execute(
                            "UPDATE thinking_gaps SET status='learned' WHERE query ILIKE %s",
                            (f"%{clean_topic}%",),
                        )
                    pg_saved = True
                finally:
                    conn.close()
            except Exception:
                pg_saved = False

        # 更新本地惊奇队列状态
        for item in self.surprise_queue:
            if item["topic"] == clean_topic:
                item["status"] = "learned"

        # 5. 记录 V20 行动证据链
        ev_pack = self.ledger.record(
            "A8_autonomous_corroborated_learning",
            clean_topic,
            {
                "source": src_name,
                "source_count": corr["source_count"],
                "corroborated": corr["corroborated"],
                "credibility": final_cred,
                "graph_new_hash": graph_delta.new_hash,
                "procedure_ir": proc_ir.ir_id,
                "pg_saved": pg_saved,
            },
        )

        record = {
            "ok": True,
            "topic": clean_topic,
            "source": src_name,
            "corroboration": corr,
            "final_credibility": final_cred,
            "brain_memory_idx": brain_idx,
            "dao_telemetry": dao_step_info,
            "graph_delta": {
                "base_hash": graph_delta.base_hash,
                "new_hash": graph_delta.new_hash,
                "added_nodes": len(graph_delta.added_nodes),
                "added_edges": len(graph_delta.added_edges),
            },
            "procedure_ir_id": proc_ir.ir_id,
            "pg_saved": pg_saved,
            "evidence_pack": ev_pack,
            "content_preview": content[:240],
        }
        self.acquisition_history.append(record)
        if len(self.acquisition_history) > 50:
            self.acquisition_history.pop(0)
        return record


import math
