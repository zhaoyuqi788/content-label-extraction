#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出模块

支持多种格式的标签文件导出：JSONL、CSV、Parquet
"""

import logging
import json
import csv
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import pandas as pd

from .schemas import ContentOutput, LabelItem, Evidence, SourceType
from .utils import load_config

logger = logging.getLogger(__name__)


class JSONLExporter:
    """JSONL格式导出器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def export(
        self, 
        content_outputs: List[ContentOutput], 
        output_path: Union[str, Path]
    ) -> bool:
        """导出为JSONL格式"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for content_output in content_outputs:
                    json_obj = self._content_output_to_dict(content_output)
                    f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                    
            logger.info(f"JSONL导出完成: {output_path}, 共 {len(content_outputs)} 条记录")
            return True
            
        except Exception as e:
            logger.error(f"JSONL导出失败: {e}")
            return False
            
    def _content_output_to_dict(self, content_output: ContentOutput) -> Dict[str, Any]:
        """将ContentOutput转换为字典"""
        result = {
            "content_id": content_output.content_id,
            "talking_angles": [self._label_item_to_dict(item) for item in content_output.talking_angles],
            "scenarios": [self._label_item_to_dict(item) for item in content_output.scenarios],
            "skin_types": [self._label_item_to_dict(item) for item in content_output.skin_types],
            "skin_concerns": [self._label_item_to_dict(item) for item in content_output.skin_concerns],
            "product_categories": [self._label_item_to_dict(item) for item in content_output.product_categories],
            "ingredients": [self._label_item_to_dict(item) for item in content_output.ingredients],
            "benefits": [self._label_item_to_dict(item) for item in content_output.benefits],
            "brands": [self._label_item_to_dict(item) for item in content_output.brands],
            "compliance_flags": [self._label_item_to_dict(item) for item in content_output.compliance_flags],
            "quality_flags": [self._label_item_to_dict(item) for item in content_output.quality_flags]
        }
        
        # 处理stance（单值）
        if content_output.stance:
            result["stance"] = self._label_item_to_dict(content_output.stance)
        else:
            result["stance"] = None
            
        # 添加备注
        if content_output.notes:
            result["notes"] = content_output.notes
            
        return result
        
    def _label_item_to_dict(self, label_item: LabelItem) -> Dict[str, Any]:
        """将LabelItem转换为字典"""
        result = {
            "label": label_item.label,
            "confidence": round(label_item.confidence, 4)
        }
        
        if label_item.evidence:
            result["evidence"] = {
                "text": label_item.evidence.text,
                "source": label_item.evidence.source.value if isinstance(label_item.evidence.source, SourceType) else str(label_item.evidence.source)
            }
            
        return result


class CSVExporter:
    """CSV格式导出器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def export(
        self, 
        content_outputs: List[ContentOutput], 
        output_path: Union[str, Path]
    ) -> bool:
        """导出为CSV格式"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为扁平化数据
            rows = []
            for content_output in content_outputs:
                row = self._content_output_to_flat_dict(content_output)
                rows.append(row)
                
            if not rows:
                logger.warning("没有数据可导出")
                return False
                
            # 获取所有字段名
            fieldnames = set()
            for row in rows:
                fieldnames.update(row.keys())
            fieldnames = sorted(fieldnames)
            
            # 写入CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                
            logger.info(f"CSV导出完成: {output_path}, 共 {len(rows)} 条记录")
            return True
            
        except Exception as e:
            logger.error(f"CSV导出失败: {e}")
            return False
            
    def _content_output_to_flat_dict(self, content_output: ContentOutput) -> Dict[str, Any]:
        """将ContentOutput转换为扁平化字典"""
        result = {
            "content_id": content_output.content_id,
            "notes": content_output.notes or ""
        }
        
        # 处理各类标签
        label_categories = [
            ("talking_angles", content_output.talking_angles),
            ("scenarios", content_output.scenarios),
            ("skin_types", content_output.skin_types),
            ("skin_concerns", content_output.skin_concerns),
            ("product_categories", content_output.product_categories),
            ("ingredients", content_output.ingredients),
            ("benefits", content_output.benefits),
            ("brands", content_output.brands),
            ("compliance_flags", content_output.compliance_flags),
            ("quality_flags", content_output.quality_flags)
        ]
        
        for category_name, labels in label_categories:
            if labels:
                # 标签列表（用分号分隔）
                result[f"{category_name}_labels"] = ";".join([label.label for label in labels])
                # 置信度列表（用分号分隔）
                result[f"{category_name}_confidences"] = ";".join([str(round(label.confidence, 4)) for label in labels])
                # 证据文本列表（用分号分隔）
                evidences = []
                sources = []
                for label in labels:
                    if label.evidence:
                        evidences.append(label.evidence.text)
                        sources.append(label.evidence.source.value if isinstance(label.evidence.source, SourceType) else str(label.evidence.source))
                    else:
                        evidences.append("")
                        sources.append("")
                result[f"{category_name}_evidences"] = ";".join(evidences)
                result[f"{category_name}_sources"] = ";".join(sources)
            else:
                result[f"{category_name}_labels"] = ""
                result[f"{category_name}_confidences"] = ""
                result[f"{category_name}_evidences"] = ""
                result[f"{category_name}_sources"] = ""
                
        # 处理stance
        if content_output.stance:
            result["stance_label"] = content_output.stance.label
            result["stance_confidence"] = round(content_output.stance.confidence, 4)
            if content_output.stance.evidence:
                result["stance_evidence"] = content_output.stance.evidence.text
                result["stance_source"] = content_output.stance.evidence.source.value if isinstance(content_output.stance.evidence.source, SourceType) else str(content_output.stance.evidence.source)
            else:
                result["stance_evidence"] = ""
                result["stance_source"] = ""
        else:
            result["stance_label"] = ""
            result["stance_confidence"] = ""
            result["stance_evidence"] = ""
            result["stance_source"] = ""
            
        return result


class ParquetExporter:
    """Parquet格式导出器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def export(
        self, 
        content_outputs: List[ContentOutput], 
        output_path: Union[str, Path]
    ) -> bool:
        """导出为Parquet格式"""
        try:
            # 检查pandas和pyarrow依赖
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError:
                logger.error("Parquet导出需要安装pyarrow: pip install pyarrow")
                return False
                
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 转换为DataFrame
            rows = []
            for content_output in content_outputs:
                # 使用嵌套结构保持数据完整性
                row = {
                    "content_id": content_output.content_id,
                    "talking_angles": [self._label_item_to_dict(item) for item in content_output.talking_angles],
                    "scenarios": [self._label_item_to_dict(item) for item in content_output.scenarios],
                    "skin_types": [self._label_item_to_dict(item) for item in content_output.skin_types],
                    "skin_concerns": [self._label_item_to_dict(item) for item in content_output.skin_concerns],
                    "product_categories": [self._label_item_to_dict(item) for item in content_output.product_categories],
                    "ingredients": [self._label_item_to_dict(item) for item in content_output.ingredients],
                    "benefits": [self._label_item_to_dict(item) for item in content_output.benefits],
                    "brands": [self._label_item_to_dict(item) for item in content_output.brands],
                    "compliance_flags": [self._label_item_to_dict(item) for item in content_output.compliance_flags],
                    "quality_flags": [self._label_item_to_dict(item) for item in content_output.quality_flags],
                    "stance": self._label_item_to_dict(content_output.stance) if content_output.stance else None,
                    "notes": content_output.notes or ""
                }
                rows.append(row)
                
            if not rows:
                logger.warning("没有数据可导出")
                return False
                
            # 创建DataFrame
            df = pd.DataFrame(rows)
            
            # 写入Parquet
            df.to_parquet(output_path, engine='pyarrow', compression='snappy')
            
            logger.info(f"Parquet导出完成: {output_path}, 共 {len(rows)} 条记录")
            return True
            
        except Exception as e:
            logger.error(f"Parquet导出失败: {e}")
            return False
            
    def _label_item_to_dict(self, label_item: LabelItem) -> Dict[str, Any]:
        """将LabelItem转换为字典"""
        result = {
            "label": label_item.label,
            "confidence": label_item.confidence
        }
        
        if label_item.evidence:
            result["evidence"] = {
                "text": label_item.evidence.text,
                "source": label_item.evidence.source.value if isinstance(label_item.evidence.source, SourceType) else str(label_item.evidence.source)
            }
        else:
            result["evidence"] = None
            
        return result


class LabelExporter:
    """标签导出器"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        self.export_config = self.config.get("export", {})
        
        # 初始化各格式导出器
        self.jsonl_exporter = JSONLExporter(self.export_config)
        self.csv_exporter = CSVExporter(self.export_config)
        self.parquet_exporter = ParquetExporter(self.export_config)
        
    def export(
        self, 
        content_outputs: List[ContentOutput],
        output_path: Union[str, Path],
        format_type: str = "jsonl"
    ) -> bool:
        """导出标签文件"""
        if not content_outputs:
            logger.warning("没有内容可导出")
            return False
            
        logger.info(f"开始导出 {len(content_outputs)} 条内容，格式: {format_type}")
        
        # 添加时间戳到文件名
        output_path = Path(output_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type.lower() == "jsonl":
            if not output_path.suffix:
                output_path = output_path.with_suffix(".jsonl")
            # 在文件名中添加时间戳
            output_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
            return self.jsonl_exporter.export(content_outputs, output_path)
            
        elif format_type.lower() == "csv":
            if not output_path.suffix:
                output_path = output_path.with_suffix(".csv")
            output_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
            return self.csv_exporter.export(content_outputs, output_path)
            
        elif format_type.lower() == "parquet":
            if not output_path.suffix:
                output_path = output_path.with_suffix(".parquet")
            output_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
            return self.parquet_exporter.export(content_outputs, output_path)
            
        else:
            logger.error(f"不支持的导出格式: {format_type}")
            return False
            
    def export_multiple_formats(
        self, 
        content_outputs: List[ContentOutput],
        output_dir: Union[str, Path],
        base_filename: str = "labels",
        formats: List[str] = None
    ) -> Dict[str, bool]:
        """导出多种格式"""
        if formats is None:
            formats = ["jsonl", "csv"]
            
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for format_type in formats:
            output_path = output_dir / f"{base_filename}.{format_type}"
            success = self.export(content_outputs, output_path, format_type)
            results[format_type] = success
            
        return results


class ExportPipeline:
    """导出流水线"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        self.pipeline_config = self.config.get("pipeline", {})
        self.exporter = LabelExporter(config_path)
        
    def process(
        self, 
        content_outputs: List[ContentOutput],
        output_path: Optional[Union[str, Path]] = None,
        format_type: Optional[str] = None
    ) -> bool:
        """处理导出"""
        # 使用配置中的默认值
        if output_path is None:
            output_path = self.pipeline_config.get("output_path", "output/labels")
            
        if format_type is None:
            format_type = self.pipeline_config.get("export_format", "jsonl")
            
        logger.info(f"开始导出处理，输出路径: {output_path}, 格式: {format_type}")
        
        try:
            success = self.exporter.export(content_outputs, output_path, format_type)
            
            if success:
                logger.info(f"导出完成，共处理 {len(content_outputs)} 条内容")
            else:
                logger.error("导出失败")
                
            return success
            
        except Exception as e:
            logger.error(f"导出处理失败: {e}")
            return False


# 便捷函数
def create_export_pipeline(config_path: Optional[str] = None) -> ExportPipeline:
    """创建导出流水线"""
    return ExportPipeline(config_path)


def export_content_outputs(
    content_outputs: List[ContentOutput],
    output_path: Union[str, Path],
    format_type: str = "jsonl",
    config_path: Optional[str] = None
) -> bool:
    """导出内容输出"""
    exporter = LabelExporter(config_path)
    return exporter.export(content_outputs, output_path, format_type)