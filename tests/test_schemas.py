#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试schemas模块"""

import unittest
from datetime import datetime
import sys
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.schemas import SourceType, Evidence, LabelItem, BrandItem, StanceItem, ContentInput, ContentLabels, ProcessingResult


class TestSourceType(unittest.TestCase):
    """测试SourceType枚举"""
    
    def test_source_type_values(self):
        """测试SourceType枚举值"""
        self.assertEqual(SourceType.TITLE.value, "title")
        self.assertEqual(SourceType.BODY.value, "body")
        self.assertEqual(SourceType.OCR.value, "ocr")
        self.assertEqual(SourceType.ASR.value, "asr")


class TestEvidence(unittest.TestCase):
    """测试Evidence类"""
    
    def test_evidence_creation(self):
        """测试Evidence对象创建"""
        evidence = Evidence(
            text="这是一个测试文本",
            source=SourceType.BODY
        )
        
        self.assertEqual(evidence.text, "这是一个测试文本")
        self.assertEqual(evidence.source, SourceType.BODY)
    
    def test_evidence_text_length_validation(self):
        """测试Evidence文本长度验证"""
        # 测试正常长度文本
        evidence = Evidence(
            text="短文本",
            source=SourceType.TITLE
        )
        
        self.assertEqual(evidence.text, "短文本")
        self.assertEqual(evidence.source, SourceType.TITLE)
        
        # 测试最大长度文本（50字符）
        long_text = "这是一个很长的文本" * 3  # 确保不超过50字符
        if len(long_text) > 50:
            long_text = long_text[:50]
        
        evidence_long = Evidence(
            text=long_text,
            source=SourceType.BODY
        )
        
        self.assertEqual(evidence_long.text, long_text)
        self.assertEqual(len(evidence_long.text), len(long_text))


class TestContentInput(unittest.TestCase):
    """测试ContentInput类"""
    
    def test_content_input_creation(self):
        """测试ContentInput对象创建"""
        content = ContentInput(
            content_id="test_001",
            title="测试标题",
            body="测试内容"
        )
        
        self.assertEqual(content.content_id, "test_001")
        self.assertEqual(content.title, "测试标题")
        self.assertEqual(content.body, "测试内容")
    
    def test_content_input_with_all_fields(self):
        """测试ContentInput所有字段"""
        content = ContentInput(
            content_id="test_002",
            title="完整测试标题",
            body="完整测试内容",
            ocr_text="OCR识别文本",
            asr_text="语音识别文本",
            extra_fields={"source": "test", "category": "beauty"}
        )
        
        self.assertEqual(content.content_id, "test_002")
        self.assertEqual(content.title, "完整测试标题")
        self.assertEqual(content.body, "完整测试内容")
        self.assertEqual(content.ocr_text, "OCR识别文本")
        self.assertEqual(content.asr_text, "语音识别文本")
        self.assertIsInstance(content.extra_fields, dict)
        self.assertEqual(content.extra_fields["source"], "test")
    
    def test_content_input_validation(self):
        """测试ContentInput验证"""
        # 测试空content_id
        with self.assertRaises(ValueError):
            ContentInput(
                content_id="",
                title="测试标题",
                body="测试内容"
            )
        
        # 测试None content_id
        with self.assertRaises(ValueError):
            ContentInput(
                content_id=None,
                title="测试标题",
                body="测试内容"
            )
    
    def test_get_combined_text(self):
        """测试获取合并文本"""
        content = ContentInput(
            content_id="test_003",
            title="测试标题",
            body="测试内容",
            ocr_text="OCR文本"
        )
        
        combined = content.get_combined_text()
        self.assertIn("测试标题", combined)
        self.assertIn("测试内容", combined)
        self.assertIn("OCR文本", combined)
    
    def test_get_text_by_source(self):
        """测试按来源获取文本"""
        content = ContentInput(
            content_id="test_004",
            title="标题文本",
            body="正文文本",
            ocr_text="OCR文本"
        )
        
        self.assertEqual(content.get_text_by_source(SourceType.TITLE), "标题文本")
        self.assertEqual(content.get_text_by_source(SourceType.BODY), "正文文本")
        self.assertEqual(content.get_text_by_source(SourceType.OCR), "OCR文本")
        self.assertIsNone(content.get_text_by_source(SourceType.ASR))


class TestLabelItem(unittest.TestCase):
    """测试LabelItem类"""
    
    def test_label_item_creation(self):
        """测试LabelItem对象创建"""
        evidence = Evidence(
            text="这是证据文本",
            source=SourceType.BODY
        )
        
        label_item = LabelItem(
            label="兰蔻",
            confidence=0.9,
            evidence=evidence
        )
        
        self.assertEqual(label_item.label, "兰蔻")
        self.assertEqual(label_item.confidence, 0.9)
        self.assertIsInstance(label_item.evidence, Evidence)
    
    def test_label_item_minimal(self):
        """测试LabelItem最小创建"""
        evidence = Evidence(
            text="简单证据",
            source=SourceType.TITLE
        )
        
        label_item = LabelItem(
            label="透明质酸",
            confidence=0.8,
            evidence=evidence
        )
        
        self.assertEqual(label_item.label, "透明质酸")
        self.assertEqual(label_item.confidence, 0.8)
        self.assertIsInstance(label_item.evidence, Evidence)
    
    def test_label_item_with_confidence_validation(self):
        """测试LabelItem置信度验证"""
        evidence = Evidence(
            text="测试证据",
            source=SourceType.BODY
        )
        
        # 测试有效置信度
        label_item = LabelItem(
            label="干性肌肤",
            confidence=0.85,
            evidence=evidence
        )
        
        self.assertEqual(label_item.label, "干性肌肤")
        self.assertEqual(label_item.confidence, 0.85)
        
        # 测试边界值
        label_item_min = LabelItem(
            label="测试标签",
            confidence=0.0,
            evidence=evidence
        )
        self.assertEqual(label_item_min.confidence, 0.0)
        
        label_item_max = LabelItem(
            label="测试标签",
            confidence=1.0,
            evidence=evidence
        )
        self.assertEqual(label_item_max.confidence, 1.0)


class TestBrandItem(unittest.TestCase):
    """测试BrandItem类"""
    
    def test_brand_item_creation(self):
        """测试BrandItem对象创建"""
        evidence = Evidence(
            text="兰蔻品牌",
            source=SourceType.BODY
        )
        
        brand_item = BrandItem(
            raw="兰蔻",
            norm_id="Lancôme",
            confidence=0.95,
            evidence=evidence
        )
        
        self.assertEqual(brand_item.raw, "兰蔻")
        self.assertEqual(brand_item.norm_id, "Lancôme")
        self.assertEqual(brand_item.confidence, 0.95)
        self.assertIsInstance(brand_item.evidence, Evidence)
    
    def test_brand_item_without_norm_id(self):
        """测试BrandItem不带标准化ID"""
        evidence = Evidence(
            text="未知品牌",
            source=SourceType.TITLE
        )
        
        brand_item = BrandItem(
            raw="未知品牌",
            confidence=0.7,
            evidence=evidence
        )
        
        self.assertEqual(brand_item.raw, "未知品牌")
        self.assertIsNone(brand_item.norm_id)
        self.assertEqual(brand_item.confidence, 0.7)


class TestStanceItem(unittest.TestCase):
    """测试StanceItem类"""
    
    def test_stance_item_creation(self):
        """测试StanceItem对象创建"""
        evidence = Evidence(
            text="这个产品很好用",
            source=SourceType.BODY
        )
        
        stance_item = StanceItem(
            label="positive",
            confidence=0.85,
            evidence=evidence
        )
        
        self.assertEqual(stance_item.label, "positive")
        self.assertEqual(stance_item.confidence, 0.85)
        self.assertIsInstance(stance_item.evidence, Evidence)
        self.assertEqual(stance_item.evidence.text, "这个产品很好用")
    
    def test_stance_item_with_confidence_validation(self):
        """测试StanceItem置信度验证"""
        evidence = Evidence(
            text="测试文本",
            source=SourceType.TITLE
        )
        
        # 测试有效置信度
        stance_item = StanceItem(
            label="neutral",
            confidence=0.5,
            evidence=evidence
        )
        self.assertEqual(stance_item.confidence, 0.5)
        
        # 测试边界值
        stance_item_min = StanceItem(
            label="negative",
            confidence=0.0,
            evidence=evidence
        )
        self.assertEqual(stance_item_min.confidence, 0.0)
        
        stance_item_max = StanceItem(
            label="positive",
            confidence=1.0,
            evidence=evidence
        )
        self.assertEqual(stance_item_max.confidence, 1.0)


class TestContentLabels(unittest.TestCase):
    """测试ContentLabels类"""
    
    def test_content_labels_creation(self):
        """测试ContentLabels对象创建"""
        # 创建测试标签
        evidence = Evidence(
            text="兰蔻精华液",
            source=SourceType.TITLE
        )
        
        talking_angle = LabelItem(
            label="推荐",
            confidence=0.9,
            evidence=evidence
        )
        
        brand_item = BrandItem(
            raw="兰蔻",
            confidence=0.95,
            evidence=evidence
        )
        
        stance_item = StanceItem(
            label="positive",
            confidence=0.85,
            evidence=evidence
        )
        
        content_labels = ContentLabels(
            content_id="test_001",
            talking_angles=[talking_angle],
            brands=[brand_item],
            stance=stance_item
        )
        
        self.assertEqual(content_labels.content_id, "test_001")
        self.assertEqual(len(content_labels.talking_angles), 1)
        self.assertEqual(content_labels.talking_angles[0].label, "推荐")
        self.assertEqual(len(content_labels.brands), 1)
        self.assertEqual(content_labels.brands[0].raw, "兰蔻")
        self.assertIsNotNone(content_labels.stance)
        self.assertEqual(content_labels.stance.label, "positive")
    
    def test_content_labels_minimal(self):
        """测试最小ContentLabels创建"""
        content_labels = ContentLabels(
            content_id="test_002"
        )
        
        self.assertEqual(content_labels.content_id, "test_002")
        self.assertEqual(len(content_labels.talking_angles), 0)
        self.assertEqual(len(content_labels.brands), 0)
        self.assertIsNone(content_labels.stance)
    
    def test_get_all_labels(self):
        """测试获取所有标签"""
        evidence = Evidence(
            text="测试证据",
            source=SourceType.BODY
        )
        
        talking_angle = LabelItem(
            label="种草",
            confidence=0.8,
            evidence=evidence
        )
        
        brand_item = BrandItem(
            raw="雅诗兰黛",
            confidence=0.9,
            evidence=evidence
        )
        
        content_labels = ContentLabels(
            content_id="test_003",
            talking_angles=[talking_angle],
            brands=[brand_item]
        )
        
        all_labels = content_labels.get_all_labels()
        
        self.assertIn('talking_angles', all_labels)
        self.assertIn('brands', all_labels)
        self.assertEqual(len(all_labels['talking_angles']), 1)
        self.assertEqual(len(all_labels['brands']), 1)


class TestProcessingResult(unittest.TestCase):
    """测试ProcessingResult类"""
    
    def test_processing_result_success(self):
        """测试成功的处理结果"""
        content_labels = ContentLabels(
            content_id="test_001"
        )
        
        result = ProcessingResult(
            content_id="test_001",
            success=True,
            labels=content_labels,
            processing_time=1.5,
            tokens_used=100
        )
        
        self.assertEqual(result.content_id, "test_001")
        self.assertTrue(result.success)
        self.assertIsNotNone(result.labels)
        self.assertEqual(result.processing_time, 1.5)
        self.assertEqual(result.tokens_used, 100)
        self.assertIsNone(result.error_message)
    
    def test_processing_result_failure(self):
        """测试失败的处理结果"""
        result = ProcessingResult(
            content_id="test_002",
            success=False,
            error_message="处理失败",
            processing_time=0.5
        )
        
        self.assertEqual(result.content_id, "test_002")
        self.assertFalse(result.success)
        self.assertIsNone(result.labels)
        self.assertEqual(result.error_message, "处理失败")
        self.assertEqual(result.processing_time, 0.5)


if __name__ == '__main__':
    unittest.main()