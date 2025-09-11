#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
归一化模块

处理品牌、成分等实体的标准化映射和别名处理
"""

import logging
import csv
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
from dataclasses import dataclass
import re
from rapidfuzz import fuzz, process

from .schemas import LabelItem, Evidence, ContentOutput
from .utils import load_config, clean_text

logger = logging.getLogger(__name__)


@dataclass
class NormalizationEntry:
    """归一化条目"""
    raw_name: str
    normalized_id: str
    normalized_name: str
    aliases: List[str]
    category: Optional[str] = None
    confidence: float = 1.0


class FuzzyMatcher:
    """模糊匹配器"""
    
    def __init__(self, threshold: float = 80.0):
        self.threshold = threshold
        
    def find_best_match(
        self, 
        query: str, 
        candidates: List[str],
        limit: int = 1
    ) -> List[Tuple[str, float]]:
        """查找最佳匹配"""
        if not query or not candidates:
            return []
            
        # 清理查询文本
        clean_query = clean_text(query).lower()
        
        # 使用rapidfuzz进行模糊匹配
        matches = process.extract(
            clean_query, 
            candidates, 
            scorer=fuzz.WRatio,
            limit=limit
        )
        
        # 过滤低于阈值的匹配
        return [(match[0], match[1]) for match in matches if match[1] >= self.threshold]
        
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """计算相似度"""
        if not text1 or not text2:
            return 0.0
            
        clean_text1 = clean_text(text1).lower()
        clean_text2 = clean_text(text2).lower()
        
        return fuzz.WRatio(clean_text1, clean_text2)


class BrandNormalizer:
    """品牌归一化器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        self.brand_dict: Dict[str, NormalizationEntry] = {}
        self.alias_to_brand: Dict[str, str] = {}
        self.fuzzy_matcher = FuzzyMatcher(threshold=85.0)
        self._load_brand_dictionary()
        
    def _load_brand_dictionary(self):
        """加载品牌词典"""
        dict_path = Path("config/synonyms_brand_aliases.csv")
        
        if not dict_path.exists():
            logger.warning(f"品牌词典文件不存在: {dict_path}")
            return
            
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    brand_name = row.get('brand_name', '').strip()
                    aliases = row.get('aliases', '').strip()
                    norm_id = row.get('normalized_id', '').strip()
                    
                    if not brand_name:
                        continue
                        
                    # 解析别名
                    alias_list = []
                    if aliases:
                        alias_list = [alias.strip() for alias in aliases.split('|') if alias.strip()]
                    
                    # 创建归一化条目
                    entry = NormalizationEntry(
                        raw_name=brand_name,
                        normalized_id=norm_id or brand_name,
                        normalized_name=brand_name,
                        aliases=alias_list
                    )
                    
                    # 建立索引
                    self.brand_dict[brand_name.lower()] = entry
                    
                    # 建立别名索引
                    for alias in alias_list:
                        self.alias_to_brand[alias.lower()] = brand_name.lower()
                        
            logger.info(f"加载品牌词典完成，共 {len(self.brand_dict)} 个品牌")
            
        except Exception as e:
            logger.error(f"加载品牌词典失败: {e}")
            
    def normalize_brand(self, brand_text: str) -> Optional[NormalizationEntry]:
        """归一化品牌名"""
        if not brand_text:
            return None
            
        clean_brand = clean_text(brand_text).lower()
        
        # 精确匹配
        if clean_brand in self.brand_dict:
            return self.brand_dict[clean_brand]
            
        # 别名匹配
        if clean_brand in self.alias_to_brand:
            main_brand = self.alias_to_brand[clean_brand]
            return self.brand_dict.get(main_brand)
            
        # 模糊匹配
        all_brands = list(self.brand_dict.keys()) + list(self.alias_to_brand.keys())
        matches = self.fuzzy_matcher.find_best_match(clean_brand, all_brands)
        
        if matches:
            matched_brand = matches[0][0]
            confidence = matches[0][1] / 100.0
            
            # 获取归一化条目
            if matched_brand in self.brand_dict:
                entry = self.brand_dict[matched_brand]
            elif matched_brand in self.alias_to_brand:
                main_brand = self.alias_to_brand[matched_brand]
                entry = self.brand_dict.get(main_brand)
            else:
                return None
                
            # 调整置信度
            if entry:
                entry = NormalizationEntry(
                    raw_name=brand_text,
                    normalized_id=entry.normalized_id,
                    normalized_name=entry.normalized_name,
                    aliases=entry.aliases,
                    confidence=confidence
                )
                
            return entry
            
        return None
        
    def normalize_brand_labels(
        self, 
        brand_labels: List[LabelItem]
    ) -> List[LabelItem]:
        """归一化品牌标签列表"""
        normalized_labels = []
        
        for label in brand_labels:
            normalized_entry = self.normalize_brand(label.label)
            
            if normalized_entry:
                # 创建归一化标签
                normalized_label = LabelItem(
                    label=normalized_entry.normalized_name,
                    confidence=label.confidence * normalized_entry.confidence,
                    evidence=label.evidence
                )
                
                # 如果有标准化ID，添加到标签中
                if hasattr(normalized_label, 'normalized_id'):
                    normalized_label.normalized_id = normalized_entry.normalized_id
                    
                normalized_labels.append(normalized_label)
            else:
                # 保留原始标签
                normalized_labels.append(label)
                
        return normalized_labels


class IngredientNormalizer:
    """成分归一化器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        self.ingredient_dict: Dict[str, NormalizationEntry] = {}
        self.alias_to_ingredient: Dict[str, str] = {}
        self.fuzzy_matcher = FuzzyMatcher(threshold=80.0)
        self._load_ingredient_dictionary()
        
    def _load_ingredient_dictionary(self):
        """加载成分词典"""
        dict_path = Path("config/ingredients_dict.csv")
        
        if not dict_path.exists():
            logger.warning(f"成分词典文件不存在: {dict_path}")
            return
            
        try:
            with open(dict_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ingredient_name = row.get('ingredient_name', '').strip()
                    aliases = row.get('aliases', '').strip()
                    category = row.get('category', '').strip()
                    benefits = row.get('benefits', '').strip()
                    
                    if not ingredient_name:
                        continue
                        
                    # 解析别名
                    alias_list = []
                    if aliases:
                        alias_list = [alias.strip() for alias in aliases.split('|') if alias.strip()]
                    
                    # 创建归一化条目
                    entry = NormalizationEntry(
                        raw_name=ingredient_name,
                        normalized_id=ingredient_name,
                        normalized_name=ingredient_name,
                        aliases=alias_list,
                        category=category
                    )
                    
                    # 建立索引
                    self.ingredient_dict[ingredient_name.lower()] = entry
                    
                    # 建立别名索引
                    for alias in alias_list:
                        self.alias_to_ingredient[alias.lower()] = ingredient_name.lower()
                        
            logger.info(f"加载成分词典完成，共 {len(self.ingredient_dict)} 个成分")
            
        except Exception as e:
            logger.error(f"加载成分词典失败: {e}")
            
    def normalize_ingredient(self, ingredient_text: str) -> Optional[NormalizationEntry]:
        """归一化成分名"""
        if not ingredient_text:
            return None
            
        clean_ingredient = clean_text(ingredient_text).lower()
        
        # 精确匹配
        if clean_ingredient in self.ingredient_dict:
            return self.ingredient_dict[clean_ingredient]
            
        # 别名匹配
        if clean_ingredient in self.alias_to_ingredient:
            main_ingredient = self.alias_to_ingredient[clean_ingredient]
            return self.ingredient_dict.get(main_ingredient)
            
        # 模糊匹配
        all_ingredients = list(self.ingredient_dict.keys()) + list(self.alias_to_ingredient.keys())
        matches = self.fuzzy_matcher.find_best_match(clean_ingredient, all_ingredients)
        
        if matches:
            matched_ingredient = matches[0][0]
            confidence = matches[0][1] / 100.0
            
            # 获取归一化条目
            if matched_ingredient in self.ingredient_dict:
                entry = self.ingredient_dict[matched_ingredient]
            elif matched_ingredient in self.alias_to_ingredient:
                main_ingredient = self.alias_to_ingredient[matched_ingredient]
                entry = self.ingredient_dict.get(main_ingredient)
            else:
                return None
                
            # 调整置信度
            if entry:
                entry = NormalizationEntry(
                    raw_name=ingredient_text,
                    normalized_id=entry.normalized_id,
                    normalized_name=entry.normalized_name,
                    aliases=entry.aliases,
                    category=entry.category,
                    confidence=confidence
                )
                
            return entry
            
        return None
        
    def normalize_ingredient_labels(
        self, 
        ingredient_labels: List[LabelItem]
    ) -> List[LabelItem]:
        """归一化成分标签列表"""
        normalized_labels = []
        
        for label in ingredient_labels:
            normalized_entry = self.normalize_ingredient(label.label)
            
            if normalized_entry:
                # 创建归一化标签
                normalized_label = LabelItem(
                    label=normalized_entry.normalized_name,
                    confidence=label.confidence * normalized_entry.confidence,
                    evidence=label.evidence
                )
                
                # 如果有分类信息，添加到标签中
                if hasattr(normalized_label, 'category'):
                    normalized_label.category = normalized_entry.category
                    
                normalized_labels.append(normalized_label)
            else:
                # 保留原始标签
                normalized_labels.append(label)
                
        return normalized_labels


class ContentNormalizer:
    """内容归一化器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        self.brand_normalizer = BrandNormalizer(config_path)
        self.ingredient_normalizer = IngredientNormalizer(config_path)
        
    def normalize_content_output(
        self, 
        content_output: ContentOutput
    ) -> ContentOutput:
        """归一化内容输出"""
        logger.info(f"开始归一化处理: {content_output.content_id}")
        
        try:
            # 创建副本
            normalized_output = ContentOutput(
                content_id=content_output.content_id,
                talking_angles=content_output.talking_angles[:],
                scenarios=content_output.scenarios[:],
                skin_types=content_output.skin_types[:],
                skin_concerns=content_output.skin_concerns[:],
                product_categories=content_output.product_categories[:],
                ingredients=self.ingredient_normalizer.normalize_ingredient_labels(
                    content_output.ingredients
                ),
                benefits=content_output.benefits[:],
                brands=self.brand_normalizer.normalize_brand_labels(
                    content_output.brands
                ),
                stance=content_output.stance,
                compliance_flags=content_output.compliance_flags[:],
                quality_flags=content_output.quality_flags[:],
                notes=content_output.notes
            )
            
            # 更新处理说明
            notes = []
            if content_output.notes:
                notes.append(content_output.notes)
            notes.append("已归一化品牌和成分")
            normalized_output.notes = "; ".join(notes)
            
            logger.info(f"归一化完成: {content_output.content_id}")
            return normalized_output
            
        except Exception as e:
            logger.error(f"归一化处理失败: {e}")
            return content_output


class NormalizationPipeline:
    """归一化流水线"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        self.normalizer = ContentNormalizer(config_path)
        
    def process(
        self, 
        content_outputs: List[ContentOutput]
    ) -> List[ContentOutput]:
        """处理归一化"""
        logger.info(f"开始批量归一化处理，共 {len(content_outputs)} 条内容")
        
        normalized_outputs = []
        
        for content_output in content_outputs:
            try:
                normalized_output = self.normalizer.normalize_content_output(
                    content_output
                )
                normalized_outputs.append(normalized_output)
                
            except Exception as e:
                logger.error(f"归一化失败 {content_output.content_id}: {e}")
                # 保留原始输出
                normalized_outputs.append(content_output)
                
        logger.info(f"批量归一化完成，成功处理 {len(normalized_outputs)} 条")
        return normalized_outputs


# 便捷函数
def create_normalization_pipeline(config_path: Optional[str] = None) -> NormalizationPipeline:
    """创建归一化流水线"""
    return NormalizationPipeline(config_path)


def normalize_content_outputs(
    content_outputs: List[ContentOutput],
    config_path: Optional[str] = None
) -> List[ContentOutput]:
    """归一化内容输出"""
    pipeline = create_normalization_pipeline(config_path)
    return pipeline.process(content_outputs)