"""数据导入模块

负责从Excel文件读取数据，进行字段映射和基础验证。
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import glob
from loguru import logger

from src.schemas import ContentInput
from src.utils import load_config


class DataIngestion:
    """数据导入器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化数据导入器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.field_mapping = self.config.get('fields_mapping', {})
        
    def scan_and_merge_excel_files(self, data_raw_dir: str = "data_raw", data_dir: str = "data") -> str:
        """扫描data_raw文件夹下所有非$开头的.xlsx文件，合并成一个parquet文件
        
        Args:
            data_raw_dir: 原始数据文件夹路径
            data_dir: 输出数据文件夹路径
            
        Returns:
            str: 生成的parquet文件路径
        """
        data_raw_path = Path(data_raw_dir)
        data_path = Path(data_dir)
        
        # 确保输出目录存在
        data_path.mkdir(exist_ok=True)
        
        # 扫描所有非$开头的.xlsx文件
        excel_files = []
        for file_path in data_raw_path.glob("*.xlsx"):
            if not file_path.name.startswith("$"):
                excel_files.append(file_path)
        
        if not excel_files:
            raise FileNotFoundError(f"在 {data_raw_dir} 文件夹中未找到任何非$开头的.xlsx文件")
        
        logger.info(f"找到 {len(excel_files)} 个Excel文件: {[f.name for f in excel_files]}")
        
        # 读取并合并所有Excel文件
        all_dataframes = []
        for file_path in excel_files:
            try:
                logger.info(f"正在读取文件: {file_path.name}")
                # 设置dtype来保持ID列的字符串格式
                dtype_dict = {'唯一ID': str}  # 保持ID列为字符串格式
                df = pd.read_excel(file_path, dtype=dtype_dict)
                # 添加源文件信息
                df['source_file'] = file_path.name
                all_dataframes.append(df)
            except Exception as e:
                logger.error(f"读取文件 {file_path.name} 失败: {e}")
                continue
        
        if not all_dataframes:
            raise ValueError("没有成功读取任何Excel文件")
        
        # 合并所有DataFrame
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"成功合并 {len(all_dataframes)} 个文件，共 {len(merged_df)} 行数据")
        
        # 生成带时间戳的parquet文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parquet_filename = f"merged_data_{timestamp}.parquet"
        parquet_path = data_path / parquet_filename
        
        # 保存为parquet文件
        merged_df.to_parquet(parquet_path, index=False)
        logger.info(f"合并数据已保存到: {parquet_path}")
        
        return str(parquet_path)
    
    def read_parquet(self, file_path: str) -> pd.DataFrame:
        """读取parquet文件
        
        Args:
            file_path: parquet文件路径
            
        Returns:
            DataFrame: 读取的数据
            
        Raises:
            FileNotFoundError: 文件不存在
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet文件不存在: {file_path}")
            
        try:
            logger.info(f"开始读取Parquet文件: {file_path}")
            df = pd.read_parquet(file_path)
            logger.info(f"成功读取 {len(df)} 行数据")
            return df
        except Exception as e:
            logger.error(f"读取Parquet文件失败: {e}")
            raise
        
    def read_excel(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """读取Excel文件
        
        Args:
            file_path: Excel文件路径
            sheet_name: 工作表名称，默认读取第一个工作表
            
        Returns:
            DataFrame: 读取的数据
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式错误
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Excel文件不存在: {file_path}")
            
        if not file_path.suffix.lower() in ['.xlsx', '.xls']:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
            
        try:
            logger.info(f"开始读取Excel文件: {file_path}")
            # 设置dtype来保持ID列的字符串格式
            dtype_dict = {'唯一ID': str}  # 保持ID列为字符串格式
            if sheet_name is None:
                # 如果没有指定sheet_name，读取第一个工作表
                df = pd.read_excel(file_path, sheet_name=0, dtype=dtype_dict)
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=dtype_dict)
            logger.info(f"成功读取 {len(df)} 行数据")
            return df
        except Exception as e:
            logger.error(f"读取Excel文件失败: {e}")
            raise
    
    def map_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """映射字段名称
        
        Args:
            df: 原始DataFrame
            
        Returns:
            DataFrame: 映射后的DataFrame
        """
        # 创建反向映射
        reverse_mapping = {v: k for k, v in self.field_mapping.items()}
        
        # 检查必需字段
        required_fields = ['content_id']
        missing_fields = []
        
        for field in required_fields:
            mapped_field = self.field_mapping.get(field, field)
            if mapped_field not in df.columns:
                missing_fields.append(mapped_field)
        
        if missing_fields:
            raise ValueError(f"缺少必需字段: {missing_fields}")
        
        # 重命名列
        df_mapped = df.rename(columns=reverse_mapping)
        
        # 确保标准字段存在
        standard_fields = ['content_id', 'title', 'body', 'ocr_text', 'asr_text']
        for field in standard_fields:
            if field not in df_mapped.columns:
                df_mapped[field] = None
        
        logger.info(f"字段映射完成，标准字段: {standard_fields}")
        return df_mapped
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证数据质量
        
        Args:
            df: 待验证的DataFrame
            
        Returns:
            DataFrame: 验证后的DataFrame
        """
        original_count = len(df)
        
        # 去除content_id为空的行
        df = df.dropna(subset=['content_id'])
        
        # 去除content_id重复的行
        df = df.drop_duplicates(subset=['content_id'], keep='first')
        
        # 转换数据类型
        df['content_id'] = df['content_id'].astype(str)
        
        # 填充空值
        text_fields = ['title', 'body', 'ocr_text', 'asr_text']
        for field in text_fields:
            df[field] = df[field].fillna('').astype(str)
        
        # 检查是否有有效文本
        def has_valid_text(row):
            return any(row[field].strip() for field in text_fields if pd.notna(row[field]))
        
        valid_mask = df.apply(has_valid_text, axis=1)
        df = df[valid_mask]
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            logger.warning(f"数据验证移除了 {removed_count} 行无效数据")
        
        logger.info(f"数据验证完成，有效数据 {len(df)} 行")
        return df
    
    def convert_to_content_inputs(self, df: pd.DataFrame) -> List[ContentInput]:
        """转换为ContentInput对象列表
        
        Args:
            df: 验证后的DataFrame
            
        Returns:
            List[ContentInput]: ContentInput对象列表
        """
        content_inputs = []
        
        for _, row in df.iterrows():
            try:
                # 提取额外字段
                standard_fields = {'content_id', 'title', 'body', 'ocr_text', 'asr_text'}
                extra_fields = {k: v for k, v in row.to_dict().items() 
                              if k not in standard_fields and pd.notna(v)}
                
                content_input = ContentInput(
                    content_id=str(row['content_id']),
                    title=row['title'] if pd.notna(row['title']) and row['title'].strip() else None,
                    body=row['body'] if pd.notna(row['body']) and row['body'].strip() else None,
                    ocr_text=row['ocr_text'] if pd.notna(row['ocr_text']) and row['ocr_text'].strip() else None,
                    asr_text=row['asr_text'] if pd.notna(row['asr_text']) and row['asr_text'].strip() else None,
                    extra_fields=extra_fields
                )
                content_inputs.append(content_input)
            except Exception as e:
                logger.error(f"转换行数据失败 content_id={row.get('content_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"成功转换 {len(content_inputs)} 个ContentInput对象")
        return content_inputs
    
    def load_data_from_merged_files(self, data_raw_dir: str = "data_raw", data_dir: str = "data") -> List[ContentInput]:
        """从data_raw文件夹合并Excel文件并加载数据的完整流程
        
        Args:
            data_raw_dir: 原始数据文件夹路径
            data_dir: 输出数据文件夹路径
            
        Returns:
            List[ContentInput]: 加载的内容列表
        """
        try:
            # 检查data文件夹中是否已有parquet文件
            data_path = Path(data_dir)
            existing_parquet_files = list(data_path.glob("*.parquet")) if data_path.exists() else []
            
            if existing_parquet_files:
                # 如果有现有的parquet文件，选择最新的一个
                latest_parquet = max(existing_parquet_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"发现已存在的parquet文件: {latest_parquet.name}，直接读取")
                parquet_path = str(latest_parquet)
            else:
                # 如果没有现有文件，则扫描并合并Excel文件生成parquet
                logger.info(f"data文件夹中未发现parquet文件，开始从{data_raw_dir}合并Excel文件")
                parquet_path = self.scan_and_merge_excel_files(data_raw_dir, data_dir)
            
            # 从parquet文件读取数据到内存
            df = self.read_parquet(parquet_path)
            
            # 步骤3: 字段映射
            df = self.map_fields(df)
            
            # 步骤4: 数据验证
            df = self.validate_data(df)
            
            # 步骤5: 转换为ContentInput
            content_inputs = self.convert_to_content_inputs(df)
            
            logger.info(f"从合并文件加载数据完成，共 {len(content_inputs)} 条有效数据")
            return content_inputs
            
        except Exception as e:
            logger.error(f"从合并文件加载数据失败: {e}")
            raise
    
    def load_data(self, file_path: str, sheet_name: Optional[str] = None) -> List[ContentInput]:
        """加载数据的完整流程
        
        Args:
            file_path: Excel文件路径
            sheet_name: 工作表名称
            
        Returns:
            List[ContentInput]: 加载的内容列表
        """
        try:
            # 读取Excel
            df = self.read_excel(file_path, sheet_name)
            
            # 字段映射
            df = self.map_fields(df)
            
            # 数据验证
            df = self.validate_data(df)
            
            # 转换为ContentInput
            content_inputs = self.convert_to_content_inputs(df)
            
            logger.info(f"数据加载完成，共 {len(content_inputs)} 条有效数据")
            return content_inputs
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    def get_data_summary(self, content_inputs: List[ContentInput]) -> Dict[str, Any]:
        """获取数据摘要统计
        
        Args:
            content_inputs: 内容列表
            
        Returns:
            Dict: 数据摘要
        """
        if not content_inputs:
            return {'total_count': 0}
        
        summary = {
            'total_count': len(content_inputs),
            'has_title': sum(1 for c in content_inputs if c.title),
            'has_body': sum(1 for c in content_inputs if c.body),
            'has_ocr': sum(1 for c in content_inputs if c.ocr_text),
            'has_asr': sum(1 for c in content_inputs if c.asr_text),
        }
        
        # 计算文本长度统计
        text_lengths = []
        for content in content_inputs:
            combined_text = content.get_combined_text()
            text_lengths.append(len(combined_text))
        
        if text_lengths:
            summary.update({
                'avg_text_length': sum(text_lengths) / len(text_lengths),
                'max_text_length': max(text_lengths),
                'min_text_length': min(text_lengths)
            })
        
        return summary


def load_excel_data(file_path: str, config_path: Optional[str] = None, 
                   sheet_name: Optional[str] = None) -> List[ContentInput]:
    """便捷函数：加载Excel数据
    
    Args:
        file_path: Excel文件路径
        config_path: 配置文件路径
        sheet_name: 工作表名称
        
    Returns:
        List[ContentInput]: 加载的内容列表
    """
    ingestion = DataIngestion(config_path)
    return ingestion.load_data(file_path, sheet_name)


def load_merged_excel_data(data_raw_dir: str = "data_raw", data_dir: str = "data", 
                          config_path: Optional[str] = None) -> List[ContentInput]:
    """便捷函数：从data_raw文件夹合并Excel文件并加载数据
    
    Args:
        data_raw_dir: 原始数据文件夹路径
        data_dir: 输出数据文件夹路径
        config_path: 配置文件路径
        
    Returns:
        List[ContentInput]: 加载的内容列表
    """
    ingestion = DataIngestion(config_path)
    return ingestion.load_data_from_merged_files(data_raw_dir, data_dir)