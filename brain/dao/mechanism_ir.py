# -*- coding: utf-8 -*-
"""
一次经验机制获取与结构迁移 (One-Shot Mechanism Acquisition & ProcedureIR)
==========================================================================

对应大兮 AGI V16/V17/V20 核心认知机制:
1. 从单次经验/思维链轨迹中抽取因果机制签名 (MechanismSignature):
   - Preconditions (前置条件与触发谓词)
   - Invariants (执行过程必须守恒的不变量)
   - Operations / ProcedureSteps (带工具绑定、验证谓词与回滚动作的可执行状态机步骤)
   - Postconditions (后置验收断言)
   - FailureSignatures & NegativeGuards (失败签名与反例收紧边界)
   - AbstractRoles (领域无关抽象角色槽位, 支持零样本/单样本跨领域结构同构映射)

2. 跨领域结构同构迁移 (StructuralTransferEngine):
   - 将源领域 (如数据库不停机迁移) 抽取到的因果机制图与角色槽位映射至目标领域 (如分布式服务灰度切流或模型热更新),
     保持因果拓扑不变并自动替换领域算子与验收探针。

3. 反例驱动边界收紧 (CounterexampleRefiner):
   - 当遇到边界失败或反例轨迹时, 精确定位违反的前置条件或不变量, 增量注入负向守卫谓词 (Negative Guard)
     与补偿分支, 零灾难性遗忘。
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import hashlib
import json
import re
import time


@dataclass
class ProcedureStep:
    step_no: int
    action: str
    rationale: str
    tool: str = "bash"
    tool_args: str = ""
    guard_predicate: str = "true"
    verify_predicate: str = "exit_code == 0"
    rollback_action: str = "restore_checkpoint"


@dataclass
class MechanismSignature:
    mechanism_id: str
    name: str
    source_domain: str
    task_type: str
    preconditions: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    failure_signatures: List[str] = field(default_factory=list)
    negative_guards: List[str] = field(default_factory=list)
    abstract_roles: Dict[str, str] = field(default_factory=dict)


@dataclass
class ProcedureIR:
    ir_id: str
    signature: MechanismSignature
    steps: List[ProcedureStep] = field(default_factory=list)
    version: int = 1
    transfer_parent_id: Optional[str] = None
    refinement_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["digest"] = self.digest()
        return d


class MechanismExtractor:
    """从单次成功案例、故障复盘或思维链五段式文本中一次性抽取因果机制与 ProcedureIR。"""

    def extract_one_shot(
        self,
        title: str,
        experience_text: str,
        source_domain: str = "software_engineering",
        steps_raw: Optional[List[str]] = None,
        tools_raw: Optional[List[Dict[str, Any]]] = None,
        pitfalls_raw: Optional[List[str]] = None,
        acceptance_raw: Optional[List[str]] = None,
    ) -> ProcedureIR:
        clean_title = (title or "未命名机制").strip()[:80]
        mech_id = "mech_" + hashlib.sha256(
            f"{source_domain}:{clean_title}:{experience_text[:200]}".encode("utf-8")
        ).hexdigest()[:12]

        # 推断任务类型
        task_type = "实现"
        for cand in ("排查", "修复", "迁移", "优化", "调研", "加固", "重构", "评估"):
            if cand in clean_title or cand in experience_text[:200]:
                task_type = cand
                break

        # 抽取抽象角色槽位 (用于跨领域同构迁移)
        roles = self._infer_abstract_roles(clean_title + " " + experience_text)

        # 构建步骤列表
        steps: List[ProcedureStep] = []
        if steps_raw:
            tool_map = {}
            for t in (tools_raw or []):
                if isinstance(t, dict):
                    s_no = int(t.get("step", len(tool_map) + 1) or (len(tool_map) + 1))
                    tool_map[s_no] = (str(t.get("tool", "bash")), str(t.get("args", "")))
            for idx, s_txt in enumerate(steps_raw, start=1):
                parts = re.split(r"\s*[—\-]{1,2}\s*", str(s_txt), maxsplit=1)
                act = re.sub(r"^\d+[\.\、\s]*", "", parts[0]).strip()
                rat = parts[1].strip() if len(parts) > 1 else "保证前置因果依赖满足"
                tl, t_args = tool_map.get(idx, ("read_file" if idx == 1 else "bash", act[:60]))
                steps.append(ProcedureStep(
                    step_no=idx,
                    action=act,
                    rationale=rat,
                    tool=tl,
                    tool_args=t_args,
                    guard_predicate=f"step_{idx - 1}_verified" if idx > 1 else "preconditions_met",
                    verify_predicate=f"verify_step_{idx}({tl})",
                    rollback_action=f"rollback_step_{idx}",
                ))
        else:
            # 从自由文本自动切分因果动作链
            sentences = [
                s.strip() for s in re.split(r"[。；;\n]+", experience_text)
                if len(s.strip()) >= 6
            ]
            for idx, sent in enumerate(sentences[:6], start=1):
                tool_guess = "grep" if any(k in sent for k in ("定位", "搜索", "排查", "查")) else (
                    "run_python" if any(k in sent for k in ("验证", "测试", "计算", "校验")) else "bash"
                )
                steps.append(ProcedureStep(
                    step_no=idx,
                    action=sent[:100],
                    rationale="单次经验因果链抽取",
                    tool=tool_guess,
                    tool_args=sent[:60],
                    guard_predicate=f"step_{idx - 1}_ok" if idx > 1 else "env_ready",
                    verify_predicate="state_consistent == True",
                    rollback_action="revert_last_mutation",
                ))

        if not steps:
            steps.append(ProcedureStep(
                step_no=1,
                action=f"执行 {clean_title} 标准基线检查与变更闭环",
                rationale="默认单步安全闭环",
                tool="bash",
                tool_args=clean_title,
            ))

        preconditions = [
            f"目标载体 [{roles.get('target_entity', 'system')}] 处于可观测状态",
            f"具备回滚快照或对偶备份通道 [{roles.get('backup_channel', 'snapshot')}]",
        ]
        invariants = [
            f"在途事务与公共契约 [{roles.get('invariant_resource', 'service_sla')}] 零中断",
            "每步状态跃迁必须通过确定性探针校验后方可进入下一步",
        ]
        if pitfalls_raw:
            failure_sigs = [str(p)[:120] for p in pitfalls_raw[:5]]
        else:
            failure_sigs = [
                f"未验证 [{roles.get('target_entity', 'target')}] 边界条件直接全量切换导致回退异常"
            ]
        if acceptance_raw:
            postconditions = [str(a)[:120] for a in acceptance_raw[:5]]
        else:
            postconditions = [
                f"[{roles.get('target_entity', 'target')}] 新状态验证通过且指标无退化",
                "审计证据链哈希完整落盘",
            ]

        sig = MechanismSignature(
            mechanism_id=mech_id,
            name=clean_title,
            source_domain=source_domain,
            task_type=task_type,
            preconditions=preconditions,
            invariants=invariants,
            postconditions=postconditions,
            failure_signatures=failure_sigs,
            negative_guards=[],
            abstract_roles=roles,
        )
        return ProcedureIR(ir_id=f"ir_{mech_id}", signature=sig, steps=steps)

    @staticmethod
    def _infer_abstract_roles(text: str) -> Dict[str, str]:
        roles = {
            "source_entity": "旧状态源组件",
            "target_entity": "新状态目标组件",
            "invariant_resource": "线上不变量契约",
            "verification_probe": "一致性校验探针",
            "backup_channel": "增量回滚通道",
        }
        if any(k in text for k in ("数据库", "表", "PG", "SQL", "schema", "迁移")):
            roles.update({
                "source_entity": "旧表结构(legacy_table)",
                "target_entity": "新表结构(shadow_table)",
                "invariant_resource": "双写事务一致性与读写延迟",
                "verification_probe": "行级哈希对账探针",
                "backup_channel": "逆向触发器与增量补偿日志",
            })
        elif any(k in text for k in ("模型", "神经网络", "神经元", "巩固", "推理", "brain")):
            roles.update({
                "source_entity": "海马体短期情景缓存(pending_episodic)",
                "target_entity": "皮层原型长期记忆库(cortical_prototypes)",
                "invariant_resource": "编码器表征稳定性与零灾难遗忘",
                "verification_probe": "余弦检索精度与良心熵探针",
                "backup_channel": "brain_v2.pkl.delta 增量回滚快照",
            })
        elif any(k in text for k in ("爬虫", "抓取", "采集", "百科", "风控")):
            roles.update({
                "source_entity": "多源外部网页候选流",
                "target_entity": "可信语料库(verified_corpus)",
                "invariant_resource": "认知免疫反污染阈值与站点冷却",
                "verification_probe": "跨源互证投票与可信度评分器",
                "backup_channel": "quarantined 隔离区与冷却黑名单",
            })
        return roles


class StructuralTransferEngine:
    """跨领域结构映射引擎: 将源领域的 ProcedureIR 零样本/单样本同构迁移至新领域。"""

    def transfer(
        self,
        source_ir: ProcedureIR,
        target_domain: str,
        role_overrides: Optional[Dict[str, str]] = None,
    ) -> ProcedureIR:
        src_roles = dict(source_ir.signature.abstract_roles)
        tgt_roles = dict(src_roles)
        if role_overrides:
            tgt_roles.update(role_overrides)
        else:
            # 根据 target_domain 自动生成目标领域角色绑定
            tgt_roles = {
                k: f"[{target_domain}]{v}" for k, v in src_roles.items()
            }

        def _rewrite(txt: str) -> str:
            out = txt
            for role_key, src_val in src_roles.items():
                tgt_val = tgt_roles.get(role_key, src_val)
                if src_val and src_val in out:
                    out = out.replace(src_val, tgt_val)
            return out

        new_mech_id = "mech_" + hashlib.sha256(
            f"{source_ir.ir_id}->transfer->{target_domain}".encode("utf-8")
        ).hexdigest()[:12]

        new_sig = MechanismSignature(
            mechanism_id=new_mech_id,
            name=f"{source_ir.signature.name} (跨域迁移 -> {target_domain})",
            source_domain=target_domain,
            task_type=source_ir.signature.task_type,
            preconditions=[_rewrite(x) for x in source_ir.signature.preconditions],
            invariants=[_rewrite(x) for x in source_ir.signature.invariants],
            postconditions=[_rewrite(x) for x in source_ir.signature.postconditions],
            failure_signatures=[_rewrite(x) for x in source_ir.signature.failure_signatures],
            negative_guards=list(source_ir.signature.negative_guards),
            abstract_roles=tgt_roles,
        )

        new_steps = [
            ProcedureStep(
                step_no=s.step_no,
                action=_rewrite(s.action),
                rationale=f"[同构迁移自 {source_ir.signature.source_domain}] {_rewrite(s.rationale)}",
                tool=s.tool,
                tool_args=_rewrite(s.tool_args),
                guard_predicate=s.guard_predicate,
                verify_predicate=s.verify_predicate,
                rollback_action=s.rollback_action,
            )
            for s in source_ir.steps
        ]

        return ProcedureIR(
            ir_id=f"ir_{new_mech_id}",
            signature=new_sig,
            steps=new_steps,
            version=source_ir.version + 1,
            transfer_parent_id=source_ir.ir_id,
            refinement_history=list(source_ir.refinement_history) + [{
                "event": "structural_transfer",
                "from_domain": source_ir.signature.source_domain,
                "to_domain": target_domain,
                "ts": time.time(),
            }],
        )


class CounterexampleRefiner:
    """反例边界收紧器: 接收失败反例轨迹, 精确收紧前置条件守卫并插入防御性验证步。"""

    def refine(
        self,
        proc_ir: ProcedureIR,
        counterexample_desc: str,
        violated_step_no: int = 1,
        guard_expression: Optional[str] = None,
    ) -> ProcedureIR:
        guard = guard_expression or f"NOT ({counterexample_desc.strip()[:80]})"
        if guard not in proc_ir.signature.negative_guards:
            proc_ir.signature.negative_guards.append(guard)
        if counterexample_desc not in proc_ir.signature.failure_signatures:
            proc_ir.signature.failure_signatures.append(counterexample_desc[:120])

        # 在对应步骤挂载强化守卫谓词
        target_idx = max(1, min(violated_step_no, len(proc_ir.steps)))
        for s in proc_ir.steps:
            if s.step_no == target_idx:
                if guard not in s.guard_predicate:
                    s.guard_predicate = f"({s.guard_predicate}) AND ({guard})"
                break

        proc_ir.version += 1
        proc_ir.refinement_history.append({
            "event": "counterexample_refinement",
            "counterexample": counterexample_desc,
            "violated_step": target_idx,
            "added_guard": guard,
            "new_version": proc_ir.version,
            "ts": time.time(),
        })
        return proc_ir
