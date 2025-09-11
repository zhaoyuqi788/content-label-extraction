#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量标签抽取主执行脚本

整合所有模块，实现完整的美妆内容标签抽取流水线
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional
import time
from datetime import datetime

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import setup_logging, load_config
from src.ingestion import load_excel_data
from src.preprocessing import create_preprocessing_pipeline
from src.rules import create_rule_engine
from src.extractor import create_extractor
from src.fusion import create_fusion_pipeline
from src.normalizer import create_normalization_pipeline
from src.exporter import create_export_pipeline
from src.schemas import ContentInput, ContentOutput

logger = logging.getLogger(__name__)


class BatchLabelingPipeline:
    """批量标签抽取流水线"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = load_config(config_path)
        
        # 初始化各个组件
        self.preprocessing_pipeline = None
        self.rule_engine = None
        self.extractor = None
        self.fusion_pipeline = None
        self.normalization_pipeline = None
        self.export_pipeline = None
        
        # 统计信息
        self.stats = {
            "total_contents": 0,
            "processed_contents": 0,
            "failed_contents": 0,
            "start_time": None,
            "end_time": None,
            "processing_time": 0
        }
        
    def initialize_components(self):
        """初始化所有组件"""
        logger.info("初始化流水线组件...")
        
        try:
            # 预处理流水线
            self.preprocessing_pipeline = create_preprocessing_pipeline(self.config_path)
            logger.info("预处理流水线初始化完成")
            
            # 规则引擎
            self.rule_engine = create_rule_engine(self.config_path)
            logger.info("规则引擎初始化完成")
            
            # LLM抽取器
            self.extractor = create_extractor(self.config_path)
            logger.info("LLM抽取器初始化完成")
            
            # 融合流水线
            self.fusion_pipeline = create_fusion_pipeline(self.config_path)
            logger.info("融合流水线初始化完成")
            
            # 归一化流水线
            self.normalization_pipeline = create_normalization_pipeline(self.config_path)
            logger.info("归一化流水线初始化完成")
            
            # 导出流水线
            self.export_pipeline = create_export_pipeline(self.config_path)
            logger.info("导出流水线初始化完成")
            
            logger.info("所有组件初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
            
    def process_batch(
        self, 
        input_file: str,
        output_path: Optional[str] = None,
        export_format: str = "jsonl"
    ) -> bool:
        """处理批量数据"""
        self.stats["start_time"] = datetime.now()
        
        try:
            # 1. 数据导入
            logger.info(f"开始导入数据: {input_file}")
            content_inputs = load_excel_data(input_file, self.config_path)
            
            if not content_inputs:
                logger.error("没有成功导入任何数据")
                return False
                
            self.stats["total_contents"] = len(content_inputs)
            logger.info(f"成功导入 {len(content_inputs)} 条内容")
            
            # 2. 预处理
            logger.info("开始预处理...")
            preprocessed_contents = self.preprocessing_pipeline.process(content_inputs)
            logger.info(f"预处理完成，处理 {len(preprocessed_contents)} 条内容")
            
            # 3. 批量抽取和融合
            logger.info("开始批量标签抽取...")
            content_outputs = []
            
            for i, content in enumerate(preprocessed_contents):
                try:
                    logger.info(f"处理第 {i+1}/{len(preprocessed_contents)} 条内容: {content.content_id}")
                    
                    # 规则抽取
                    rule_results = self.rule_engine.extract_labels([content])
                    
                    # LLM抽取
                    llm_results = self.extractor.extract_batch([content])
                    
                    # 融合结果
                    if rule_results or llm_results:
                        fused_output = self.fusion_pipeline.process(rule_results, llm_results)
                        content_outputs.append(fused_output)
                        self.stats["processed_contents"] += 1
                    else:
                        logger.warning(f"内容 {content.content_id} 没有抽取到任何标签")
                        # 创建空输出
                        empty_output = ContentOutput(
                            content_id=content.content_id,
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
                            notes="无抽取结果"
                        )
                        content_outputs.append(empty_output)
                        
                except Exception as e:
                    logger.error(f"处理内容 {content.content_id} 失败: {e}")
                    self.stats["failed_contents"] += 1
                    continue
                    
            if not content_outputs:
                logger.error("没有成功处理任何内容")
                return False
                
            logger.info(f"标签抽取完成，成功处理 {len(content_outputs)} 条内容")
            
            # 4. 归一化
            logger.info("开始归一化处理...")
            normalized_outputs = self.normalization_pipeline.process(content_outputs)
            logger.info(f"归一化完成，处理 {len(normalized_outputs)} 条内容")
            
            # 5. 导出结果
            logger.info(f"开始导出结果，格式: {export_format}")
            if output_path is None:
                output_path = f"output/labels_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
            success = self.export_pipeline.process(
                normalized_outputs, 
                output_path, 
                export_format
            )
            
            if success:
                logger.info(f"导出完成: {output_path}")
            else:
                logger.error("导出失败")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            return False
            
        finally:
            self.stats["end_time"] = datetime.now()
            if self.stats["start_time"]:
                self.stats["processing_time"] = (
                    self.stats["end_time"] - self.stats["start_time"]
                ).total_seconds()
                
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "="*50)
        print("批量处理统计信息")
        print("="*50)
        print(f"总内容数: {self.stats['total_contents']}")
        print(f"成功处理: {self.stats['processed_contents']}")
        print(f"处理失败: {self.stats['failed_contents']}")
        
        if self.stats['total_contents'] > 0:
            success_rate = (self.stats['processed_contents'] / self.stats['total_contents']) * 100
            print(f"成功率: {success_rate:.2f}%")
            
        if self.stats['processing_time'] > 0:
            print(f"处理时间: {self.stats['processing_time']:.2f} 秒")
            if self.stats['processed_contents'] > 0:
                avg_time = self.stats['processing_time'] / self.stats['processed_contents']
                print(f"平均每条: {avg_time:.2f} 秒")
                
        print("="*50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="美妆内容批量标签抽取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_batch.py -i data_raw/content.xlsx -o output/labels -f jsonl
  python run_batch.py -i data_raw/content.xlsx -f csv
  python run_batch.py -i data_raw/content.xlsx --config config/config.yaml
        """
    )
    
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="输入Excel文件路径"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="输出文件路径（不含扩展名）"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["jsonl", "csv", "parquet"],
        default="jsonl",
        help="输出格式 (默认: jsonl)"
    )
    
    parser.add_argument(
        "-c", "--config",
        help="配置文件路径 (默认: config/config.yaml)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="日志级别 (默认: INFO)"
    )
    
    parser.add_argument(
        "--log-file",
        help="日志文件路径"
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(
        level=getattr(logging, args.log_level),
        log_file=args.log_file
    )
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)
        
    # 检查配置文件
    config_path = args.config or "config/config.yaml"
    if not Path(config_path).exists():
        logger.warning(f"配置文件不存在: {config_path}，将使用默认配置")
        config_path = None
        
    logger.info("开始批量标签抽取任务")
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出格式: {args.format}")
    if args.output:
        logger.info(f"输出路径: {args.output}")
    if config_path:
        logger.info(f"配置文件: {config_path}")
        
    try:
        # 创建流水线
        pipeline = BatchLabelingPipeline(config_path)
        
        # 初始化组件
        pipeline.initialize_components()
        
        # 处理批量数据
        success = pipeline.process_batch(
            input_file=args.input,
            output_path=args.output,
            export_format=args.format
        )
        
        # 打印统计信息
        pipeline.print_stats()
        
        if success:
            logger.info("批量处理完成")
            sys.exit(0)
        else:
            logger.error("批量处理失败")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("用户中断处理")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()