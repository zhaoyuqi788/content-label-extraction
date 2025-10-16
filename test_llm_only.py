#!/usr/bin/env python3
"""
只测试LLM抽取功能，跳过融合阶段
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger
from src.ingestion import load_merged_excel_data
from src.preprocessing import create_preprocessing_pipeline
from src.extractor import LabelExtractor
from src.utils import load_config

async def test_llm_extraction_only():
    """只测试LLM抽取功能"""
    
    try:
        # 加载配置
        config = load_config()
        
        # 1. 数据导入（只取1条数据）
        logger.info("开始导入数据...")
        content_inputs = load_merged_excel_data("data_raw", "data", "config/config.yaml")
        
        if not content_inputs:
            logger.error("没有成功导入任何数据")
            return False
            
        logger.info(f"成功导入 {len(content_inputs)} 条内容")
        
        # 2. 预处理（只处理1条数据）
        logger.info("开始预处理...")
        preprocessing_pipeline = create_preprocessing_pipeline("config/config.yaml")
        test_contents = content_inputs[:1]  # 只测试1条
        preprocessed_contents = preprocessing_pipeline.process(test_contents)
        logger.info(f"预处理完成，处理 {len(preprocessed_contents)} 条内容")
        
        # 3. 初始化LLM抽取器
        logger.info("初始化LLM抽取器...")
        extractor = LabelExtractor("config/config.yaml")
        logger.info("LLM抽取器初始化完成")
        
        # 4. LLM抽取（跳过规则引擎）
        logger.info("开始LLM抽取...")
        llm_results = await extractor.extract_batch(preprocessed_contents)
        logger.info(f"LLM抽取完成，结果数量: {len(llm_results)}")
        
        # 5. 输出结果信息
        for result in llm_results:
            if result.success:
                logger.info(f"内容 {result.content_id} LLM抽取成功")
                logger.info(f"  - 处理时间: {result.processing_time:.2f}秒")
                logger.info(f"  - Token使用: {result.tokens_used}")
                
                # 显示抽取到的标签数量
                labels = result.labels
                if labels:
                    logger.info(f"  - 抽取标签统计:")
                    logger.info(f"    * talking_angles: {len(labels.talking_angles)}")
                    logger.info(f"    * scenarios: {len(labels.scenarios)}")
                    logger.info(f"    * skin_types: {len(labels.skin_types)}")
                    logger.info(f"    * skin_concerns: {len(labels.skin_concerns)}")
                    logger.info(f"    * product_categories: {len(labels.product_categories)}")
                    logger.info(f"    * ingredients: {len(labels.ingredients)}")
                    logger.info(f"    * benefits: {len(labels.benefits)}")
                    logger.info(f"    * brands: {len(labels.brands)}")
                    logger.info(f"    * stance: {labels.stance}")
                    
                    # 显示一些具体标签
                    if labels.talking_angles:
                        logger.info(f"    * talking_angles示例: {[item.label for item in labels.talking_angles[:3]]}")
                    if labels.brands:
                        logger.info(f"    * brands示例: {[item.raw for item in labels.brands[:3]]}")
            else:
                logger.error(f"内容 {result.content_id} LLM抽取失败: {result.error_message}")
        
        logger.info("LLM抽取测试完成")
        return True
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_llm_extraction_only())
    if success:
        logger.info("✅ LLM抽取测试成功")
        sys.exit(0)
    else:
        logger.error("❌ LLM抽取测试失败")
        sys.exit(1)