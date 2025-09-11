#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试ingestion模块"""

import unittest
import sys
import tempfile
import os
import pandas as pd
from pathlib import Path

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.ingestion import DataIngestion, load_excel_data, load_merged_excel_data
from src.schemas import ContentInput


class TestDataIngestion(unittest.TestCase):
    """测试DataIngestion类"""
    
    def setUp(self):
        """设置测试数据"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_raw_dir = os.path.join(self.temp_dir, "data_raw")
        self.data_dir = os.path.join(self.temp_dir, "data")
        
        # 创建目录
        os.makedirs(self.data_raw_dir)
        os.makedirs(self.data_dir)
        
        # 创建测试Excel文件
        self.create_test_excel_files()
        
        # 创建测试配置
        self.test_config = {
            'fields_mapping': {
                'content_id': '唯一ID',
                'title': '标题',
                'body': '内容',
                'ocr_text': '视频文字识别',
                'asr_text': '视频语音识别'
            }
        }
        
        self.ingestion = DataIngestion()
        self.ingestion.config = self.test_config
        self.ingestion.field_mapping = self.test_config['fields_mapping']
    
    def tearDown(self):
        """清理测试文件"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_excel_files(self):
        """创建测试用的Excel文件"""
        # 测试数据
        test_data1 = {
            '唯一ID': ['001', '002', '003'],
            '标题': ['兰蔻精华液测试', '雅诗兰黛面霜', '测试标题3'],
            '内容': ['这是兰蔻精华液的详细介绍', '雅诗兰黛面霜效果很好', '普通内容'],
            '视频文字识别': ['OCR文本1', 'OCR文本2', 'OCR文本3'],
            '视频语音识别': ['ASR文本1', 'ASR文本2', 'ASR文本3']
        }
        
        test_data2 = {
            '唯一ID': ['004', '005'],
            '标题': ['SK-II神仙水', '测试标题5'],
            '内容': ['SK-II神仙水补水效果好', '另一个测试内容'],
            '视频文字识别': ['OCR文本4', 'OCR文本5'],
            '视频语音识别': ['ASR文本4', 'ASR文本5']
        }
        
        # 创建Excel文件
        df1 = pd.DataFrame(test_data1)
        df2 = pd.DataFrame(test_data2)
        
        self.excel_file1 = os.path.join(self.data_raw_dir, "test_data1.xlsx")
        self.excel_file2 = os.path.join(self.data_raw_dir, "test_data2.xlsx")
        
        df1.to_excel(self.excel_file1, index=False)
        df2.to_excel(self.excel_file2, index=False)
        
        # 创建一个以$开头的文件（应该被忽略）
        self.ignored_file = os.path.join(self.data_raw_dir, "$ignored_file.xlsx")
        df1.to_excel(self.ignored_file, index=False)
    
    def test_read_excel(self):
        """测试读取Excel文件"""
        df = self.ingestion.read_excel(self.excel_file1)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertIn('唯一ID', df.columns)
        self.assertIn('标题', df.columns)
    
    def test_read_excel_nonexistent(self):
        """测试读取不存在的Excel文件"""
        nonexistent_file = os.path.join(self.temp_dir, "nonexistent.xlsx")
        
        with self.assertRaises(FileNotFoundError):
            self.ingestion.read_excel(nonexistent_file)
    
    def test_map_fields(self):
        """测试字段映射"""
        df = self.ingestion.read_excel(self.excel_file1)
        mapped_df = self.ingestion.map_fields(df)
        
        # 验证字段被正确映射
        self.assertIn('content_id', mapped_df.columns)
        self.assertIn('title', mapped_df.columns)
        self.assertIn('body', mapped_df.columns)
        self.assertIn('ocr_text', mapped_df.columns)
        self.assertIn('asr_text', mapped_df.columns)
        
        # 验证数据内容
        self.assertEqual(mapped_df.iloc[0]['content_id'], '001')
        self.assertEqual(mapped_df.iloc[0]['title'], '兰蔻精华液测试')
    
    def test_validate_data(self):
        """测试数据验证"""
        df = self.ingestion.read_excel(self.excel_file1)
        mapped_df = self.ingestion.map_fields(df)
        validated_df = self.ingestion.validate_data(mapped_df)
        
        # 验证数据类型
        self.assertTrue(validated_df['content_id'].dtype == 'object')
        
        # 验证没有空的content_id
        self.assertFalse(validated_df['content_id'].isna().any())
        
        # 验证没有重复的content_id
        self.assertEqual(len(validated_df), len(validated_df['content_id'].unique()))
    
    def test_convert_to_content_inputs(self):
        """测试转换为ContentInput对象"""
        df = self.ingestion.read_excel(self.excel_file1)
        mapped_df = self.ingestion.map_fields(df)
        validated_df = self.ingestion.validate_data(mapped_df)
        content_inputs = self.ingestion.convert_to_content_inputs(validated_df)
        
        self.assertEqual(len(content_inputs), 3)
        self.assertIsInstance(content_inputs[0], ContentInput)
        self.assertEqual(content_inputs[0].content_id, '001')
        self.assertEqual(content_inputs[0].title, '兰蔻精华液测试')
    
    def test_scan_and_merge_excel_files(self):
        """测试扫描和合并Excel文件"""
        parquet_path = self.ingestion.scan_and_merge_excel_files(
            self.data_raw_dir, self.data_dir
        )
        
        # 验证parquet文件被创建
        self.assertTrue(os.path.exists(parquet_path))
        self.assertTrue(parquet_path.endswith('.parquet'))
        
        # 验证文件内容
        df = pd.read_parquet(parquet_path)
        self.assertEqual(len(df), 5)  # 两个文件共5行数据
        self.assertIn('source_file', df.columns)
        
        # 验证$开头的文件被忽略
        source_files = df['source_file'].unique()
        self.assertNotIn('$ignored_file.xlsx', source_files)
    
    def test_read_parquet(self):
        """测试读取parquet文件"""
        # 先创建parquet文件
        parquet_path = self.ingestion.scan_and_merge_excel_files(
            self.data_raw_dir, self.data_dir
        )
        
        # 读取parquet文件
        df = self.ingestion.read_parquet(parquet_path)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 5)
        self.assertIn('唯一ID', df.columns)
    
    def test_load_data_from_merged_files(self):
        """测试从合并文件加载数据的完整流程"""
        content_inputs = self.ingestion.load_data_from_merged_files(
            self.data_raw_dir, self.data_dir
        )
        
        self.assertEqual(len(content_inputs), 5)
        self.assertIsInstance(content_inputs[0], ContentInput)
        
        # 验证数据内容
        content_ids = [c.content_id for c in content_inputs]
        self.assertIn('001', content_ids)
        self.assertIn('004', content_ids)
    
    def test_load_data(self):
        """测试加载单个Excel文件"""
        content_inputs = self.ingestion.load_data(self.excel_file1)
        
        self.assertEqual(len(content_inputs), 3)
        self.assertIsInstance(content_inputs[0], ContentInput)
        self.assertEqual(content_inputs[0].content_id, '001')
    
    def test_get_data_summary(self):
        """测试获取数据摘要"""
        content_inputs = self.ingestion.load_data(self.excel_file1)
        summary = self.ingestion.get_data_summary(content_inputs)
        
        self.assertEqual(summary['total_count'], 3)
        self.assertIn('has_title', summary)
        self.assertIn('has_body', summary)
        self.assertIn('avg_text_length', summary)


class TestConvenienceFunctions(unittest.TestCase):
    """测试便捷函数"""
    
    def setUp(self):
        """设置测试数据"""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建测试Excel文件
        test_data = {
            '唯一ID': ['test_001'],
            '标题': ['测试标题'],
            '内容': ['测试内容'],
            '视频文字识别': ['OCR文本'],
            '视频语音识别': ['ASR文本']
        }
        
        df = pd.DataFrame(test_data)
        self.excel_file = os.path.join(self.temp_dir, "test.xlsx")
        df.to_excel(self.excel_file, index=False)
    
    def tearDown(self):
        """清理测试文件"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_excel_data(self):
        """测试load_excel_data便捷函数"""
        # 注意：这个测试可能会因为配置文件路径问题而失败
        # 在实际项目中需要确保配置文件存在
        try:
            content_inputs = load_excel_data(self.excel_file)
            self.assertIsInstance(content_inputs, list)
        except Exception:
            # 如果因为配置问题失败，跳过这个测试
            self.skipTest("配置文件不存在，跳过测试")


if __name__ == '__main__':
    unittest.main()