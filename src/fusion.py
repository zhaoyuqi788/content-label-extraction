#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融合与冲突解决模块

处理多来源标签融合、置信度计算和冲突消解
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import numpy as np
from copy import deepcopy

from .schemas import (
    LabelItem, Evidence, ContentOutput, SourceType,
    ExtractionResult, FusionConfig, ContentLabels, BrandItem, ContentInput
)
from .utils import load_config

logger = logging.getLogger(__name__)


@dataclass
class ConflictRule:
    """冲突规则定义"""
    category: str
    mutually_exclusive: List[List[str]] = field(default_factory=list)
    priority_order: List[str] = field(default_factory=list)
    max_labels: Optional[int] = None
    

@dataclass
class FusionMetrics:
    """融合指标"""
    total_labels: int = 0
    fused_labels: int = 0
    conflicts_resolved: int = 0
    high_confidence: int = 0
    review_needed: int = 0
    discarded: int = 0
    

class ConfidenceFuser:
    """置信度融合器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.source_weights = {
            SourceType.BODY: 1.0,
            SourceType.TITLE: 0.9,
            SourceType.ASR: 0.8,
            SourceType.OCR: 0.7
        }
        
    def fuse_confidences(
        self, 
        confidences: List[float], 
        sources: List[SourceType],
        method: str = "weighted_max"
    ) -> float:
        """融合多个置信度"""
        if not confidences:
            return 0.0
            
        if len(confidences) == 1:
            return confidences[0]
            
        if method == "weighted_max":
            # 加权最大值
            weighted_scores = [
                conf * self.source_weights.get(src, 0.5) 
                for conf, src in zip(confidences, sources)
            ]
            return min(max(weighted_scores), 1.0)
            
        elif method == "weighted_mean":
            # 加权平均
            weights = [self.source_weights.get(src, 0.5) for src in sources]
            weighted_sum = sum(c * w for c, w in zip(confidences, weights))
            weight_sum = sum(weights)
            return min(weighted_sum / weight_sum if weight_sum > 0 else 0.0, 1.0)
            
        elif method == "evidence_boost":
            # 多证据提升
            base_conf = max(confidences)
            boost = min(0.1 * (len(confidences) - 1), 0.3)
            return min(base_conf + boost, 1.0)
            
        else:
            return max(confidences)
            
    def calculate_evidence_strength(
        self, 
        evidences: List[Evidence]
    ) -> float:
        """计算证据强度"""
        if not evidences:
            return 0.0
            
        # 基于证据长度、来源多样性等计算强度
        source_diversity = len(set(ev.source for ev in evidences))
        avg_length = np.mean([len(ev.text) for ev in evidences])
        
        # 归一化到0-1
        diversity_score = min(source_diversity / 4.0, 1.0)  # 最多4个来源
        length_score = min(avg_length / 50.0, 1.0)  # 50字符为满分
        
        return (diversity_score + length_score) / 2.0


class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.conflict_rules = self._load_conflict_rules()
        
    def _load_conflict_rules(self) -> Dict[str, ConflictRule]:
        """加载冲突规则"""
        rules = {
            "stance": ConflictRule(
                category="stance",
                mutually_exclusive=[
                    ["强种草", "强烈拔草"],
                    ["一般种草", "轻微拔草"]
                ],
                max_labels=1
            ),
            "skin_types": ConflictRule(
                category="skin_types",
                max_labels=3  # 最多3种肤质
            ),
            "talking_angles": ConflictRule(
                category="talking_angles",
                max_labels=3  # 最多3个角度
            )
        }
        return rules
        
    def resolve_conflicts(
        self, 
        labels: List[LabelItem], 
        category: str
    ) -> List[LabelItem]:
        """解决标签冲突"""
        if category not in self.conflict_rules:
            return labels
            
        rule = self.conflict_rules[category]
        resolved_labels = list(labels)
        
        # 处理互斥标签
        for exclusive_group in rule.mutually_exclusive:
            resolved_labels = self._resolve_exclusive_conflict(
                resolved_labels, exclusive_group
            )
            
        # 处理数量限制
        if rule.max_labels and len(resolved_labels) > rule.max_labels:
            resolved_labels = sorted(
                resolved_labels, 
                key=lambda x: x.confidence, 
                reverse=True
            )[:rule.max_labels]
            
        return resolved_labels
        
    def _resolve_exclusive_conflict(
        self, 
        labels: List[LabelItem], 
        exclusive_group: List[str]
    ) -> List[LabelItem]:
        """解决互斥冲突"""
        conflicting_labels = [
            label for label in labels 
            if label.label in exclusive_group
        ]
        
        if len(conflicting_labels) <= 1:
            return labels
            
        # 选择置信度最高的
        best_label = max(conflicting_labels, key=lambda x: x.confidence)
        
        # 移除其他冲突标签
        non_conflicting = [
            label for label in labels 
            if label.label not in exclusive_group
        ]
        
        return non_conflicting + [best_label]


class LabelFuser:
    """标签融合器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.confidence_fuser = ConfidenceFuser(config)
        self.conflict_resolver = ConflictResolver(config)
        self.accept_threshold = config.get("accept_threshold", 0.75)
        self.review_threshold = config.get("review_threshold", 0.4)
        
    def fuse_labels(
        self, 
        rule_results: List[ExtractionResult],
        llm_results: List[ExtractionResult]
    ) -> ContentOutput:
        """融合规则和LLM结果"""
        all_results = rule_results + llm_results
        
        if not all_results:
            return ContentOutput(
                content_id="",
                talking_angles=[],
                scenarios=[],
                skin_types=[],
                skin_concerns=[],
                product_categories=[],
                ingredients=[],
                benefits=[],
                brands=[],
                stance=None,
                compliance_flags=[],
                quality_flags=[],
                notes="No extraction results"
            )
            
        # 按类别分组标签
        grouped_labels = self._group_labels_by_category(all_results)
        
        # 创建融合后的ContentLabels
        fused_labels = ContentLabels(
            content_id=all_results[0].content_id,
            talking_angles=self._fuse_category_labels(
                grouped_labels.get("talking_angles", []), "talking_angles"
            ),
            scenarios=self._fuse_category_labels(
                grouped_labels.get("scenarios", []), "scenarios"
            ),
            skin_types=self._fuse_category_labels(
                grouped_labels.get("skin_types", []), "skin_types"
            ),
            skin_concerns=self._fuse_category_labels(
                grouped_labels.get("skin_concerns", []), "skin_concerns"
            ),
            product_categories=self._fuse_category_labels(
                grouped_labels.get("product_categories", []), "product_categories"
            ),
            ingredients=self._fuse_category_labels(
                grouped_labels.get("ingredients", []), "ingredients"
            ),
            benefits=self._fuse_category_labels(
                grouped_labels.get("benefits", []), "benefits"
            ),
            brands=self._fuse_brand_labels(
                grouped_labels.get("brands", [])
            ),
            compliance_flags=self._fuse_category_labels(
                grouped_labels.get("compliance_flags", []), "compliance_flags"
            ),
            quality_flags=self._fuse_category_labels(
                grouped_labels.get("quality_flags", []), "quality_flags"
            )
        )
        
        # 处理stance（单值）
        stance_labels = grouped_labels.get("stance", [])
        if stance_labels:
            fused_stance = self._fuse_category_labels(stance_labels, "stance")
            fused_labels.stance = fused_stance[0] if fused_stance else None
        
        # 创建ContentOutput对象
        # 创建一个基本的ContentInput对象作为original_content
        original_content = ContentInput(
            content_id=all_results[0].content_id,
            title="",  # 实际应用中应该从原始数据获取
            body="",
            ocr_text="",
            asr_text=""
        )
        
        fused_output = ContentOutput(
            content_id=all_results[0].content_id,
            original_content=original_content,
            labels=fused_labels,
            processing_metadata={
                "fusion_method": "weighted_voting",
                "num_sources": len(all_results),
                "fusion_timestamp": "2024-01-01T00:00:00Z"  # 临时时间戳
            }
        )
            
        return fused_output
        
    def _group_labels_by_category(
        self, 
        results: List[ExtractionResult]
    ) -> Dict[str, List[LabelItem]]:
        """按类别分组标签"""
        grouped = defaultdict(list)
        
        for result in results:
            # 从result.labels中获取标签
            labels_obj = result.labels
            for attr_name in [
                "talking_angles", "scenarios", "skin_types", "skin_concerns",
                "product_categories", "ingredients", "benefits", "brands",
                "compliance_flags", "quality_flags"
            ]:
                labels = getattr(labels_obj, attr_name, [])
                if labels:
                    grouped[attr_name].extend(labels)
                    
            # 处理stance
            if hasattr(labels_obj, 'stance') and labels_obj.stance:
                grouped["stance"].append(labels_obj.stance)
                
        return grouped
        
    def _fuse_category_labels(
        self, 
        labels: List[LabelItem], 
        category: str
    ) -> List[LabelItem]:
        """融合单个类别的标签"""
        if not labels:
            return []
            
        # 按标签名分组
        label_groups = defaultdict(list)
        for label in labels:
            label_groups[label.label].append(label)
            
        # 融合同名标签
        fused_labels = []
        for label_name, label_list in label_groups.items():
            fused_label = self._fuse_same_labels(label_list)
            if fused_label.confidence >= self.review_threshold:
                fused_labels.append(fused_label)
                
        # 解决冲突
        resolved_labels = self.conflict_resolver.resolve_conflicts(
            fused_labels, category
        )
        
        # 按置信度排序
        return sorted(resolved_labels, key=lambda x: x.confidence, reverse=True)
    
    def _fuse_brand_labels(self, labels: List[LabelItem]) -> List[BrandItem]:
        """融合品牌标签，转换为BrandItem格式"""
        if not labels:
            return []
            
        # 按标签名分组
        label_groups = defaultdict(list)
        for label in labels:
            label_groups[label.label].append(label)
            
        # 融合同名标签并转换为BrandItem
        fused_brands = []
        for label_name, label_list in label_groups.items():
            fused_label = self._fuse_same_labels(label_list)
            if fused_label.confidence >= self.review_threshold:
                # 从extra_data中获取品牌信息，如果没有则使用默认值
                extra_data = getattr(fused_label, 'extra_data', {}) or {}
                raw_name = extra_data.get('raw', fused_label.label)
                norm_id = extra_data.get('norm_id')
                
                brand_item = BrandItem(
                    raw=raw_name,
                    norm_id=norm_id,
                    confidence=fused_label.confidence,
                    evidence=fused_label.evidence
                )
                fused_brands.append(brand_item)
                
        # 按置信度排序
        return sorted(fused_brands, key=lambda x: x.confidence, reverse=True)
        
    def _fuse_same_labels(self, labels: List[LabelItem]) -> LabelItem:
        """融合相同标签"""
        if len(labels) == 1:
            return labels[0]
            
        # 收集所有置信度和证据
        confidences = [label.confidence for label in labels]
        all_evidences = []
        sources = []
        
        for label in labels:
            if label.evidence:
                all_evidences.append(label.evidence)
                sources.append(label.evidence.source)
                
        # 融合置信度
        fused_confidence = self.confidence_fuser.fuse_confidences(
            confidences, sources
        )
        
        # 选择最佳证据（优先级最高且文本最长）
        best_evidence = None
        if all_evidences:
            source_priority = {
                SourceType.BODY: 4,
                SourceType.TITLE: 3,
                SourceType.ASR: 2,
                SourceType.OCR: 1
            }
            
            best_evidence = max(
                all_evidences,
                key=lambda ev: (source_priority.get(ev.source, 0), len(ev.text))
            )
            
        return LabelItem(
            label=labels[0].label,
            confidence=fused_confidence,
            evidence=best_evidence
        )


class FusionPipeline:
    """融合流水线"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        self.pipeline_config = self.config.get("pipeline", {})
        self.fuser = LabelFuser(self.pipeline_config)
        self.metrics = FusionMetrics()
        
    def process(
        self, 
        rule_results: List[ExtractionResult],
        llm_results: List[ExtractionResult]
    ) -> ContentOutput:
        """处理融合"""
        logger.info(f"开始融合处理，规则结果: {len(rule_results)}, LLM结果: {len(llm_results)}")
        
        try:
            # 融合标签
            output = self.fuser.fuse_labels(rule_results, llm_results)
            
            # 更新指标
            self._update_metrics(output)
            
            # 添加处理说明
            notes = []
            if rule_results:
                notes.append(f"规则匹配: {len(rule_results)}条")
            if llm_results:
                notes.append(f"LLM抽取: {len(llm_results)}条")
                
            output.labels.notes = "; ".join(notes) if notes else "无抽取结果"
            
            logger.info(f"融合完成，输出标签总数: {self._count_total_labels(output)}")
            return output
            
        except Exception as e:
            logger.error(f"融合处理失败: {e}")
            raise
            
    def _update_metrics(self, output: ContentOutput):
        """更新指标"""
        total_labels = self._count_total_labels(output)
        self.metrics.total_labels += total_labels
        
        # 统计不同置信度区间的标签
        all_labels = self._get_all_labels(output)
        for label in all_labels:
            if label.confidence >= self.fuser.accept_threshold:
                self.metrics.high_confidence += 1
            elif label.confidence >= self.fuser.review_threshold:
                self.metrics.review_needed += 1
            else:
                self.metrics.discarded += 1
                
    def _count_total_labels(self, output: ContentOutput) -> int:
        """统计总标签数"""
        count = 0
        for attr_name in [
            "talking_angles", "scenarios", "skin_types", "skin_concerns",
            "product_categories", "ingredients", "benefits", "brands",
            "compliance_flags", "quality_flags"
        ]:
            labels = getattr(output.labels, attr_name, [])
            count += len(labels) if labels else 0
            
        if output.labels.stance:
            count += 1
            
        return count
        
    def _get_all_labels(self, output: ContentOutput) -> List[LabelItem]:
        """获取所有标签"""
        all_labels = []
        
        for attr_name in [
            "talking_angles", "scenarios", "skin_types", "skin_concerns",
            "product_categories", "ingredients", "benefits", "brands",
            "compliance_flags", "quality_flags"
        ]:
            labels = getattr(output.labels, attr_name, [])
            if labels:
                all_labels.extend(labels)
                
        if output.labels.stance:
            all_labels.append(output.labels.stance)
            
        return all_labels
        
    def get_metrics(self) -> FusionMetrics:
        """获取融合指标"""
        return self.metrics
        
    def reset_metrics(self):
        """重置指标"""
        self.metrics = FusionMetrics()


# 便捷函数
def create_fusion_pipeline(config_path: Optional[str] = None) -> FusionPipeline:
    """创建融合流水线"""
    return FusionPipeline(config_path)


def fuse_extraction_results(
    rule_results: List[ExtractionResult],
    llm_results: List[ExtractionResult],
    config_path: Optional[str] = None
) -> ContentOutput:
    """融合抽取结果"""
    pipeline = create_fusion_pipeline(config_path)
    return pipeline.process(rule_results, llm_results)