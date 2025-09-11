"""标签抽取器模块

整合规则引擎和LLM抽取，提供完整的标签抽取流程。
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from loguru import logger

from .schemas import (
    ContentInput, ContentLabels, ProcessingResult, 
    LLMExtractionRequest, SourceType, Evidence, LabelItem
)
from .llm_client import LLMClientManager, BatchProcessor
from .prompts import PromptTemplate, PromptOptimizer
from .utils import load_config, setup_logging, truncate_text


class LabelExtractor:
    """标签抽取器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化标签抽取器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        setup_logging(self.config)
        
        # 初始化组件
        self.llm_client = LLMClientManager(self.config['model'])
        self.prompt_template = PromptTemplate()
        self.prompt_optimizer = PromptOptimizer()
        
        # 处理配置
        pipeline_config = self.config.get('pipeline', {})
        self.max_chars = pipeline_config.get('max_chars_per_segment', 1500)
        self.accept_threshold = pipeline_config.get('accept_threshold', 0.75)
        self.review_threshold = pipeline_config.get('review_threshold', 0.4)
        self.source_priority = pipeline_config.get('source_priority', ['body', 'title', 'asr', 'ocr'])
        
        logger.info("标签抽取器初始化完成")
    
    async def extract_single(self, content: ContentInput) -> ProcessingResult:
        """提取单个内容的标签
        
        Args:
            content: 内容输入
            
        Returns:
            ProcessingResult: 处理结果
        """
        start_time = time.time()
        
        try:
            logger.info(f"开始处理内容: {content.content_id}")
            
            # 预处理文本
            processed_content = self._preprocess_content(content)
            
            # 构建抽取请求
            extraction_request = self._build_extraction_request(processed_content)
            
            # LLM抽取
            llm_response = await self.llm_client.extract_labels(extraction_request)
            
            # 后处理和验证
            final_labels = self._postprocess_labels(llm_response.extracted_labels)
            
            processing_time = time.time() - start_time
            
            # 记录性能
            self.prompt_optimizer.record_performance(
                content.content_id,
                processing_time,
                self._calculate_quality_score(final_labels),
                llm_response.tokens_used
            )
            
            logger.info(f"完成处理内容: {content.content_id}, 耗时: {processing_time:.2f}s")
            
            return ProcessingResult(
                content_id=content.content_id,
                success=True,
                labels=final_labels,
                processing_time=processing_time,
                tokens_used=llm_response.tokens_used
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"处理内容失败 {content.content_id}: {e}")
            
            return ProcessingResult(
                content_id=content.content_id,
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    async def extract_batch(self, contents: List[ContentInput]) -> List[ProcessingResult]:
        """批量提取标签
        
        Args:
            contents: 内容列表
            
        Returns:
            List[ProcessingResult]: 处理结果列表
        """
        logger.info(f"开始批量处理 {len(contents)} 个内容")
        
        # 构建批量请求
        requests = []
        for content in contents:
            try:
                processed_content = self._preprocess_content(content)
                request = self._build_extraction_request(processed_content)
                requests.append(request)
            except Exception as e:
                logger.error(f"预处理失败 {content.content_id}: {e}")
                # 添加错误结果
                requests.append(None)
        
        # 批量处理
        batch_processor = BatchProcessor(
            self.llm_client,
            self.config['pipeline'].get('batch_size', 16)
        )
        
        valid_requests = [r for r in requests if r is not None]
        llm_responses = await batch_processor.process_batch(valid_requests)
        
        # 构建最终结果
        results = []
        llm_idx = 0
        
        for i, content in enumerate(contents):
            if requests[i] is None:
                # 预处理失败的情况
                results.append(ProcessingResult(
                    content_id=content.content_id,
                    success=False,
                    error_message="预处理失败"
                ))
            else:
                # 正常处理的情况
                llm_response = llm_responses[llm_idx]
                llm_idx += 1
                
                try:
                    final_labels = self._postprocess_labels(llm_response.extracted_labels)
                    
                    results.append(ProcessingResult(
                        content_id=content.content_id,
                        success=True,
                        labels=final_labels,
                        processing_time=llm_response.processing_time,
                        tokens_used=llm_response.tokens_used
                    ))
                except Exception as e:
                    logger.error(f"后处理失败 {content.content_id}: {e}")
                    results.append(ProcessingResult(
                        content_id=content.content_id,
                        success=False,
                        error_message=f"后处理失败: {e}",
                        processing_time=llm_response.processing_time,
                        tokens_used=llm_response.tokens_used
                    ))
        
        logger.info(f"批量处理完成，成功: {sum(1 for r in results if r.success)}/{len(results)}")
        return results
    
    def _preprocess_content(self, content: ContentInput) -> ContentInput:
        """预处理内容
        
        Args:
            content: 原始内容
            
        Returns:
            ContentInput: 预处理后的内容
        """
        # 截断过长文本
        processed_content = ContentInput(
            content_id=content.content_id,
            title=truncate_text(content.title, 200) if content.title else None,
            body=truncate_text(content.body, self.max_chars) if content.body else None,
            ocr_text=truncate_text(content.ocr_text, 500) if content.ocr_text else None,
            asr_text=truncate_text(content.asr_text, 500) if content.asr_text else None,
            extra_fields=content.extra_fields
        )
        
        # 检查是否有有效文本
        combined_text = processed_content.get_combined_text()
        if not combined_text.strip():
            raise ValueError("没有有效的文本内容")
        
        # 记录截断信息
        original_length = len(content.get_combined_text())
        processed_length = len(combined_text)
        
        if processed_length < original_length:
            logger.warning(f"内容被截断 {content.content_id}: {original_length} -> {processed_length}")
        
        return processed_content
    
    def _build_extraction_request(self, content: ContentInput) -> LLMExtractionRequest:
        """构建抽取请求
        
        Args:
            content: 预处理后的内容
            
        Returns:
            LLMExtractionRequest: 抽取请求
        """
        # 构建提示文本
        system_prompt = self.prompt_template.get_system_prompt()
        user_prompt = self.prompt_template.get_user_prompt(content)
        
        # 优化提示
        optimized_prompt = self.prompt_optimizer.optimize_prompt_for_content(
            content, user_prompt
        )
        
        # 合并提示
        full_prompt = f"{system_prompt}\n\n{optimized_prompt}"
        
        # 构建来源映射
        source_mapping = {}
        if content.title:
            source_mapping['title'] = SourceType.TITLE
        if content.body:
            source_mapping['body'] = SourceType.BODY
        if content.ocr_text:
            source_mapping['ocr'] = SourceType.OCR
        if content.asr_text:
            source_mapping['asr'] = SourceType.ASR
        
        return LLMExtractionRequest(
            content_id=content.content_id,
            text=full_prompt,
            source_mapping=source_mapping
        )
    
    def _postprocess_labels(self, labels: ContentLabels) -> ContentLabels:
        """后处理标签
        
        Args:
            labels: 原始标签
            
        Returns:
            ContentLabels: 处理后的标签
        """
        # 过滤低置信度标签
        filtered_labels = ContentLabels(content_id=labels.content_id)
        
        # 处理各类标签
        for category, items in labels.get_all_labels().items():
            if category == 'stance' and items:
                # 情感立场是单个项目
                item = items[0]
                if item.confidence >= self.review_threshold:
                    filtered_labels.stance = item
            else:
                # 其他是列表项目
                filtered_items = [
                    item for item in items 
                    if item.confidence >= self.review_threshold
                ]
                
                # 按置信度排序
                filtered_items.sort(key=lambda x: x.confidence, reverse=True)
                
                # 设置到对应字段
                setattr(filtered_labels, category, filtered_items)
        
        # 添加质量标识
        quality_flags = self._detect_quality_issues(labels)
        filtered_labels.quality_flags.extend(quality_flags)
        
        return filtered_labels
    
    def _detect_quality_issues(self, labels: ContentLabels) -> List[LabelItem]:
        """检测质量问题
        
        Args:
            labels: 标签对象
            
        Returns:
            List[LabelItem]: 质量问题标签
        """
        quality_issues = []
        
        # 检查是否有足够的标签
        total_labels = sum(len(getattr(labels, attr, [])) for attr in [
            'talking_angles', 'scenarios', 'skin_types', 'skin_concerns',
            'product_categories', 'ingredients', 'benefits'
        ])
        
        if total_labels < 3:
            quality_issues.append(LabelItem(
                label="标签过少",
                confidence=0.8,
                evidence=Evidence(
                    text=f"总标签数: {total_labels}",
                    source=SourceType.BODY
                )
            ))
        
        # 检查置信度分布
        all_confidences = []
        for category, items in labels.get_all_labels().items():
            if category != 'stance':
                all_confidences.extend([item.confidence for item in items])
            elif items:
                all_confidences.append(items[0].confidence)
        
        if all_confidences:
            avg_confidence = sum(all_confidences) / len(all_confidences)
            if avg_confidence < 0.6:
                quality_issues.append(LabelItem(
                    label="整体置信度偏低",
                    confidence=0.7,
                    evidence=Evidence(
                        text=f"平均置信度: {avg_confidence:.2f}",
                        source=SourceType.BODY
                    )
                ))
        
        return quality_issues
    
    def _calculate_quality_score(self, labels: ContentLabels) -> float:
        """计算质量评分
        
        Args:
            labels: 标签对象
            
        Returns:
            float: 质量评分 (0-1)
        """
        scores = []
        
        # 标签数量评分
        total_labels = sum(len(getattr(labels, attr, [])) for attr in [
            'talking_angles', 'scenarios', 'skin_types', 'skin_concerns',
            'product_categories', 'ingredients', 'benefits'
        ])
        
        label_score = min(1.0, total_labels / 10.0)  # 10个标签为满分
        scores.append(label_score)
        
        # 置信度评分
        all_confidences = []
        for category, items in labels.get_all_labels().items():
            if category != 'stance':
                all_confidences.extend([item.confidence for item in items])
            elif items:
                all_confidences.append(items[0].confidence)
        
        if all_confidences:
            confidence_score = sum(all_confidences) / len(all_confidences)
            scores.append(confidence_score)
        
        # 质量问题扣分
        quality_penalty = len(labels.quality_flags) * 0.1
        
        final_score = sum(scores) / len(scores) if scores else 0.0
        final_score = max(0.0, final_score - quality_penalty)
        
        return final_score
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计
        
        Returns:
            Dict: 性能统计数据
        """
        return self.prompt_optimizer.get_performance_stats()


class ExtractionPipeline:
    """抽取流水线"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化抽取流水线
        
        Args:
            config_path: 配置文件路径
        """
        self.extractor = LabelExtractor(config_path)
        self.config = self.extractor.config
    
    async def run(self, contents: List[ContentInput]) -> List[ProcessingResult]:
        """运行抽取流水线
        
        Args:
            contents: 内容列表
            
        Returns:
            List[ProcessingResult]: 处理结果列表
        """
        logger.info(f"启动抽取流水线，处理 {len(contents)} 个内容")
        
        start_time = time.time()
        
        # 批量抽取
        results = await self.extractor.extract_batch(contents)
        
        total_time = time.time() - start_time
        
        # 统计结果
        success_count = sum(1 for r in results if r.success)
        total_tokens = sum(r.tokens_used or 0 for r in results)
        
        logger.info(f"抽取流水线完成: 成功 {success_count}/{len(results)}, "
                   f"总耗时 {total_time:.2f}s, 总tokens {total_tokens}")
        
        return results


def create_extractor(config_path: Optional[str] = None) -> LabelExtractor:
    """便捷函数：创建标签抽取器
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        LabelExtractor: 抽取器实例
    """
    return LabelExtractor(config_path)


async def extract_labels_from_contents(contents: List[ContentInput], 
                                     config_path: Optional[str] = None) -> List[ProcessingResult]:
    """便捷函数：从内容列表提取标签
    
    Args:
        contents: 内容列表
        config_path: 配置文件路径
        
    Returns:
        List[ProcessingResult]: 处理结果列表
    """
    pipeline = ExtractionPipeline(config_path)
    return await pipeline.run(contents)