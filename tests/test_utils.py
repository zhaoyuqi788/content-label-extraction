#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试utils模块"""

import unittest
import sys
from pathlib import Path
import tempfile
import os
import yaml
import json

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.utils import load_config, setup_logging, clean_text, truncate_text, extract_evidence, calculate_confidence_fusion, is_chinese_text, normalize_brand_name


class TestConfigUtils(unittest.TestCase):
    """测试配置相关工具函数"""
    
    def setUp(self):
        """设置测试数据"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_data = {
            "model": {
                "provider": "dashscope",
                "model_name": "qwen-plus",
                "temperature": 0.2
            },
            "pipeline": {
                "batch_size": 16,
                "max_chars_per_segment": 1500
            }
        }
    
    def tearDown(self):
        """清理测试文件"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_config_yaml(self):
        """测试加载YAML配置文件"""
        config_file = os.path.join(self.temp_dir, "config.yaml")
        
        # 创建测试配置文件
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config_data, f, allow_unicode=True)
        
        # 加载配置
        loaded_config = load_config(config_file)
        
        self.assertEqual(loaded_config["model"]["provider"], "dashscope")
        self.assertEqual(loaded_config["model"]["model_name"], "qwen-plus")
        self.assertEqual(loaded_config["pipeline"]["batch_size"], 16)
    
    def test_load_config_json(self):
        """测试加载JSON配置文件"""
        config_file = os.path.join(self.temp_dir, "config.json")
        
        # 创建测试配置文件
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config_data, f, ensure_ascii=False, indent=2)
        
        # 加载配置
        loaded_config = load_config(config_file)
        
        self.assertEqual(loaded_config["model"]["provider"], "dashscope")
        self.assertEqual(loaded_config["pipeline"]["batch_size"], 16)
    
    def test_load_config_nonexistent(self):
        """测试加载不存在的配置文件"""
        nonexistent_file = os.path.join(self.temp_dir, "nonexistent.yaml")
        
        with self.assertRaises(FileNotFoundError):
            load_config(nonexistent_file)
    
    def test_load_config_default(self):
        """测试加载默认配置"""
        # 当传入None时，应该加载默认配置
        config = load_config(None)
        self.assertIsInstance(config, dict)
        # 默认配置应该包含基本字段
        self.assertIn("model", config)
        self.assertIn("pipeline", config)
    



class TestTextPreprocessing(unittest.TestCase):
    """测试文本预处理函数"""
    
    def test_clean_text_basic(self):
        """测试基本文本清理"""
        text = "  这是一个测试文本！！！  \n\n  包含多余空格和换行。  "
        processed = clean_text(text)
        
        # 验证去除了多余空格和换行
        self.assertNotIn("\n", processed)
        self.assertFalse(processed.startswith(" "))
        self.assertFalse(processed.endswith(" "))
        self.assertIn("测试文本", processed)
    
    def test_clean_text_empty(self):
        """测试空文本清理"""
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text("   "), "")
    
    def test_clean_text_html(self):
        """测试HTML标签清理"""
        text = "<p>这是<strong>HTML</strong>文本</p>"
        processed = clean_text(text)
        
        # 验证HTML标签被移除
        self.assertNotIn("<p>", processed)
        self.assertNotIn("<strong>", processed)
        self.assertNotIn("</p>", processed)
        self.assertIn("这是HTML文本", processed)
    
    def test_truncate_text(self):
        """测试文本截断"""
        # 测试短文本
        short_text = "这是短文本"
        result = truncate_text(short_text, 100)
        self.assertEqual(result, short_text)
        
        # 测试长文本截断
        long_text = "这是一个很长的文本。" * 100
        result = truncate_text(long_text, 50)
        self.assertLessEqual(len(result), 53)  # 考虑省略号
        
        # 测试在句号处截断
        text_with_period = "第一句话。第二句话。第三句话。" * 20
        result = truncate_text(text_with_period, 30)
        self.assertTrue(result.endswith('。') or result.endswith('...'))


class TestUtilityFunctions(unittest.TestCase):
    """测试工具函数"""
    
    def test_extract_evidence(self):
        """测试证据提取"""
        text = "这是一段包含兰蔻精华液的长文本，用于测试证据提取功能。"
        keyword = "兰蔻"
        
        evidence = extract_evidence(text, keyword, 20)
        self.assertIn(keyword, evidence)
        self.assertLessEqual(len(evidence), 25)  # 考虑省略号
        
        # 测试关键词不存在
        evidence = extract_evidence(text, "不存在的词", 20)
        self.assertLessEqual(len(evidence), 20)
    
    def test_calculate_confidence_fusion(self):
        """测试置信度融合"""
        confidences = [0.8, 0.6, 0.9, 0.7]
        
        # 测试最大值方法
        result = calculate_confidence_fusion(confidences, "max")
        self.assertEqual(result, 0.9)
        
        # 测试平均值方法
        result = calculate_confidence_fusion(confidences, "average")
        self.assertEqual(result, 0.75)
        
        # 测试加权最大值方法
        result = calculate_confidence_fusion(confidences, "weighted_max")
        self.assertGreater(result, 0.7)
        self.assertLessEqual(result, 1.0)
        
        # 测试空列表
        result = calculate_confidence_fusion([], "max")
        self.assertEqual(result, 0.0)
    
    def test_is_chinese_text(self):
        """测试中文文本判断"""
        # 测试中文文本
        chinese_text = "这是中文文本"
        self.assertTrue(is_chinese_text(chinese_text))
        
        # 测试英文文本
        english_text = "This is English text"
        self.assertFalse(is_chinese_text(english_text))
        
        # 测试混合文本
        mixed_text = "这是mixed文本with English"
        result = is_chinese_text(mixed_text)
        # 结果取决于中文字符比例
        self.assertIsInstance(result, bool)
        
        # 测试空文本
        self.assertFalse(is_chinese_text(""))
        self.assertFalse(is_chinese_text(None))
    
    def test_normalize_brand_name(self):
        """测试品牌名称标准化"""
        # 测试基本标准化
        brand = "Lancôme"
        normalized = normalize_brand_name(brand)
        self.assertEqual(normalized, "lancôme")
        
        # 测试中文品牌
        brand = "兰蔻 (Lancôme)"
        normalized = normalize_brand_name(brand)
        self.assertEqual(normalized, "兰蔻lancôme")
        
        # 测试空字符串
        self.assertEqual(normalize_brand_name(""), "")
        self.assertEqual(normalize_brand_name(None), "")


class TestLoggingSetup(unittest.TestCase):
    """测试日志设置函数"""
    
    def test_setup_logging_basic(self):
        """测试基本日志设置"""
        # 测试不会抛出异常
        test_config = {
            'logging': {
                'level': 'INFO',
                'format': '{time} | {level} | {message}'
            }
        }
        try:
            setup_logging(test_config)
        except Exception as e:
            self.fail(f"setup_logging raised an exception: {e}")
    
    def test_setup_logging_with_file(self):
        """测试带文件输出的日志设置"""
        test_config = {
            'logging': {
                'level': 'DEBUG',
                'format': '{time} | {level} | {message}',
                'rotation': '1 MB',
                'retention': '1 day'
            }
        }
        
        try:
            setup_logging(test_config)
            
            # 验证日志文件被创建（如果有日志输出的话）
            # 这里只测试函数调用不会出错
        except Exception as e:
            self.fail(f"setup_logging with file raised an exception: {e}")
    
    def test_setup_logging_invalid_level(self):
        """测试无效日志级别"""
        # 测试无效的日志级别是否会被正确处理
        test_config = {
            'logging': {
                'level': 'INVALID_LEVEL'
            }
        }
        try:
            setup_logging(test_config)
        except Exception:
            # 如果抛出异常，这是预期的行为
            pass


if __name__ == '__main__':
    unittest.main()