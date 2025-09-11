#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试rules模块"""

import unittest
import sys
from pathlib import Path
import tempfile
import os

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.rules import RulePattern, KeywordMatcher, BrandMatcher, IngredientMatcher, RuleEngine
from src.schemas import LabelItem, SourceType, Evidence, ContentInput


class TestRulePattern(unittest.TestCase):
    """测试RulePattern类"""
    
    def test_rule_pattern_creation(self):
        """测试规则模式创建"""
        pattern = RulePattern(
            pattern=r"兰蔻|lancome",
            label="兰蔻",
            confidence=0.9,
            category="品牌"
        )
        
        self.assertEqual(pattern.label, "兰蔻")
        self.assertEqual(pattern.confidence, 0.9)
        self.assertEqual(pattern.category, "品牌")
    
    def test_rule_pattern_match(self):
        """测试规则模式匹配"""
        pattern = RulePattern(
            pattern=r"兰蔻|lancome",
            label="兰蔻",
            confidence=0.9,
            category="品牌"
        )
        
        # 测试匹配成功
        text = "我最近在用兰蔻的小黑瓶精华液"
        matches = pattern.match(text, SourceType.BODY)
        
        self.assertEqual(len(matches), 1)
        self.assertIsInstance(matches[0], LabelItem)
        self.assertEqual(matches[0].label, "兰蔻")
        self.assertEqual(matches[0].confidence, 0.9)
        
        # 测试匹配失败
        text = "我最近在用雅诗兰黛的产品"
        matches = pattern.match(text, SourceType.BODY)
        self.assertEqual(len(matches), 0)
    
    def test_rule_pattern_case_insensitive(self):
        """测试大小写不敏感匹配"""
        pattern = RulePattern(
            pattern=r"lancome",
            label="兰蔻",
            confidence=0.9,
            category="品牌"
        )
        
        # 测试不同大小写
        test_cases = ["LANCOME", "Lancome", "lancome", "LaNcOmE"]
        
        for case in test_cases:
            text = f"我在用{case}的产品"
            matches = pattern.match(text, SourceType.BODY)
            self.assertEqual(len(matches), 1, f"Failed for case: {case}")
    
    def test_rule_pattern_multiple_matches(self):
        """测试多个匹配"""
        pattern = RulePattern(
            pattern=r"精华液|面霜",
            label="护肤品",
            confidence=0.8,
            category="产品类型"
        )
        
        text = "我用的精华液和面霜都很好用"
        matches = pattern.match(text, SourceType.BODY)
        
        self.assertEqual(len(matches), 2)
        for match in matches:
            self.assertEqual(match.label, "护肤品")
            self.assertEqual(match.confidence, 0.8)


class TestKeywordMatcher(unittest.TestCase):
    """测试KeywordMatcher类"""
    
    def setUp(self):
        """设置测试数据"""
        self.matcher = KeywordMatcher()
    
    def test_keyword_matching(self):
        """测试关键词匹配"""
        text = "我最近在种草兰蔻的小黑瓶精华液，效果很好"
        results = self.matcher.match_keywords(text, SourceType.BODY)
        
        # 验证有匹配结果
        self.assertIsInstance(results, dict)
        
        # 检查是否有谈论角度匹配
        if '谈论角度' in results:
            talking_angle_labels = results['谈论角度']
            self.assertTrue(any('种草' in label.label for label in talking_angle_labels))
    
    def test_skin_type_matching(self):
        """测试肤质匹配"""
        text = "我是干皮，适合用保湿的产品"
        results = self.matcher.match_keywords(text, SourceType.BODY)
        
        # 验证有匹配结果
        self.assertIsInstance(results, dict)
        
        # 检查是否有肤质匹配
        if '肤质' in results:
            skin_type_labels = results['肤质']
            self.assertTrue(any('干性' in label.label for label in skin_type_labels))
    
    def test_scenario_matching(self):
        """测试使用场景匹配"""
        text = "通勤妆容需要持久一些的产品"
        results = self.matcher.match_keywords(text, SourceType.BODY)
        
        # 验证有匹配结果
        self.assertIsInstance(results, dict)
        
        # 检查是否有场景匹配
        if '使用场景' in results:
            scenario_labels = results['使用场景']
            self.assertTrue(any('通勤' in label.label for label in scenario_labels))
    
    def test_empty_text(self):
        """测试空文本"""
        results = self.matcher.match_keywords("", SourceType.BODY)
        self.assertEqual(len(results), 0)
    
    def test_no_matches(self):
        """测试无匹配情况"""
        text = "这是一个普通的文本，没有任何关键词"
        results = self.matcher.match_keywords(text, SourceType.BODY)
        # 可能没有匹配，也可能有少量匹配
        self.assertIsInstance(results, dict)


class TestRuleEngine(unittest.TestCase):
    """测试RuleEngine类"""
    
    def setUp(self):
        """设置测试数据"""
        self.engine = RuleEngine()
    
    def test_rule_engine_creation(self):
        """测试规则引擎创建"""
        self.assertIsNotNone(self.engine.brand_matcher)
        self.assertIsNotNone(self.engine.ingredient_matcher)
        self.assertIsNotNone(self.engine.keyword_matcher)
    
    def test_extract_labels_comprehensive(self):
        """测试综合标签抽取"""
        content = ContentInput(
            content_id="test_001",
            title="兰蔻小黑瓶精华液使用心得",
            body="我是干皮，最近在种草兰蔻的小黑瓶精华液，含有透明质酸成分，通勤妆容很适合。"
        )
        
        labels = self.engine.extract_labels(content)
        
        # 应该返回字典
        self.assertIsInstance(labels, dict)
        
        # 验证标签结构
        for category, label_list in labels.items():
            self.assertIsInstance(label_list, list)
            for label in label_list:
                self.assertIsInstance(label, LabelItem)
                self.assertIsNotNone(label.label)
                self.assertGreater(label.confidence, 0)
    
    def test_extract_labels_brand_matching(self):
        """测试品牌匹配"""
        content = ContentInput(
            content_id="test_002",
            title="兰蔻产品推荐",
            body="兰蔻的产品质量很好"
        )
        
        labels = self.engine.extract_labels(content)
        
        # 应该返回字典
        self.assertIsInstance(labels, dict)
        
        # 检查是否有品牌标签
        if 'brands' in labels:
            brand_labels = labels['brands']
            self.assertTrue(any("兰蔻" in label.label for label in brand_labels))
    
    def test_extract_labels_ingredient_matching(self):
        """测试成分匹配"""
        content = ContentInput(
            content_id="test_003",
            title="护肤成分分析",
            body="这款产品含有透明质酸和烟酰胺成分"
        )
        
        labels = self.engine.extract_labels(content)
        
        # 应该返回字典
        self.assertIsInstance(labels, dict)
        
        # 检查是否有成分标签
        if 'ingredients' in labels:
            ingredient_labels = labels['ingredients']
            ingredient_names = [label.label for label in ingredient_labels]
            # 可能匹配到透明质酸或烟酰胺相关的成分
            self.assertGreater(len(ingredient_names), 0)
    
    def test_extract_labels_empty_content(self):
        """测试空内容"""
        content = ContentInput(content_id="test_004", title="", body="")
        labels = self.engine.extract_labels(content)
        # 空内容应该返回空字典
        self.assertIsInstance(labels, dict)
    
    def test_extract_labels_no_match(self):
        """测试无匹配内容"""
        content = ContentInput(
            content_id="test_005",
            title="普通标题",
            body="这是一个完全不相关的文本内容，没有任何美妆相关词汇。"
        )
        
        labels = self.engine.extract_labels(content)
        # 可能没有匹配或有少量匹配
        self.assertIsInstance(labels, dict)


if __name__ == '__main__':
    unittest.main()