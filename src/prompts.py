"""提示工程模块

定义用于LLM标签抽取的系统提示和用户提示模板。
"""

from typing import Dict, List, Any
from .schemas import ContentInput
from .utils import load_taxonomy


class PromptTemplate:
    """提示模板类"""
    
    def __init__(self, taxonomy_path: str = None):
        """初始化提示模板
        
        Args:
            taxonomy_path: 标签分类体系文件路径
        """
        self.taxonomy = load_taxonomy(taxonomy_path)
        
    def get_system_prompt(self) -> str:
        """获取系统提示
        
        Returns:
            str: 系统提示文本
        """
        return """你是美妆内容标注助手。任务：从给定内容（标题/正文/OCR/ASR）中抽取预定义标签。

要求：
1. 仅使用给定的枚举标签，未提及的标签返回空数组
2. 返回严格JSON格式，字段顺序不限，但必须符合Schema
3. 为每个标签提供置信度(0-1)与证据文本(≤50字)和来源(title/body/ocr/asr)
4. 置信度评估标准：
   - 0.9-1.0: 明确直接提及
   - 0.7-0.9: 强烈暗示或间接提及
   - 0.5-0.7: 可能相关但不确定
   - 0.3-0.5: 弱相关
   - 0.0-0.3: 几乎无关或不确定
5. 证据文本应包含关键词或相关描述，优先选择最有说服力的片段
6. 来源标识：title(标题)、body(正文)、ocr(图片文字)、asr(视频语音)
7. 对于口语化表达，需要理解其真实含义：
   - "大干皮"→干性肌肤
   - "踩雷"→拔草/负面评价
   - "无限回购"→种草/正面推荐
   - "烂脸"→痘痘肌或严重肌肤问题
8. 情感立场判断：
   - 强种草：明确推荐、无限回购、真香等
   - 强烈拔草：踩雷、避雷、翻车、不推荐等
   - 中性：客观描述、测评对比等
9. 合规检测重点：
   - 广告标识：合作、广告、种草官等
   - 夸大功效：根治、永久、100%等绝对化词汇
   - 医疗化用语：治疗、处方、药用等
"""
    
    def get_taxonomy_prompt(self) -> str:
        """获取标签分类体系提示
        
        Returns:
            str: 分类体系文本
        """
        taxonomy_text = "标签分类体系：\n"
        
        for category, labels in self.taxonomy.items():
            if category == 'slang_mappings':  # 跳过口语映射
                continue
            taxonomy_text += f"{category}: {labels}\n"
        
        return taxonomy_text
    
    def get_user_prompt(self, content: ContentInput) -> str:
        """获取用户提示
        
        Args:
            content: 内容输入对象
            
        Returns:
            str: 用户提示文本
        """
        # 构建内容文本
        content_parts = []
        
        if content.title:
            content_parts.append(f'title: "{content.title}"')
        if content.body:
            content_parts.append(f'body: "{content.body}"')
        if content.ocr_text:
            content_parts.append(f'ocr_text: "{content.ocr_text}"')
        if content.asr_text:
            content_parts.append(f'asr_text: "{content.asr_text}"')
        
        content_text = "\n".join(content_parts)
        
        prompt = f"""{self.get_taxonomy_prompt()}

content:
content_id: {content.content_id}
{content_text}

请输出符合以下JSON Schema的对象：
{{
  "content_id": "{content.content_id}",
  "talking_angles": [
    {{"label": "标签名", "confidence": 0.0-1.0, "evidence": {{"text": "证据文本≤50字", "source": "title|body|ocr|asr"}}}}
  ],
  "scenarios": [...],
  "skin_types": [...],
  "skin_concerns": [...],
  "product_categories": [...],
  "ingredients": [...],
  "benefits": [...],
  "brands": [
    {{"raw": "原始品牌名", "norm_id": null, "confidence": 0.0-1.0, "evidence": {{"text": "证据文本", "source": "来源"}}}}
  ],
  "stance": {{"label": "情感立场", "confidence": 0.0-1.0, "evidence": {{"text": "证据文本", "source": "来源"}}}},
  "compliance_flags": [...],
  "quality_flags": [...],
  "notes": "可选备注"
}}
"""
        return prompt
    
    def get_conflict_resolution_prompt(self, content_id: str, conflicts: List[Dict[str, Any]]) -> str:
        """获取冲突解决提示
        
        Args:
            content_id: 内容ID
            conflicts: 冲突列表
            
        Returns:
            str: 冲突解决提示
        """
        conflicts_text = "\n".join([
            f"- {conflict['category']}: {conflict['labels']} (置信度: {conflict['confidences']}, 证据: {conflict['evidences']})"
            for conflict in conflicts
        ])
        
        return f"""检测到以下标签冲突，请基于证据强度和文本内容进行裁决：

content_id: {content_id}
冲突项：
{conflicts_text}

请返回解决后的标签选择，格式：
{{
  "resolved_labels": [
    {{"category": "类别", "label": "最终标签", "confidence": 0.0-1.0, "reason": "选择理由"}}
  ]
}}
"""
    
    def get_quality_check_prompt(self, content: ContentInput, extracted_labels: Dict[str, Any]) -> str:
        """获取质量检查提示
        
        Args:
            content: 内容输入
            extracted_labels: 已提取的标签
            
        Returns:
            str: 质量检查提示
        """
        return f"""请对以下标签抽取结果进行质量检查：

原始内容：
{content.get_combined_text()[:500]}...

抽取结果：
{extracted_labels}

检查项目：
1. 标签是否与内容匹配
2. 置信度是否合理
3. 证据文本是否准确
4. 是否存在遗漏的重要标签
5. 是否存在错误标签

请返回质量评估：
{{
  "quality_score": 0.0-1.0,
  "issues": [
    {{"type": "问题类型", "description": "问题描述", "severity": "high|medium|low"}}
  ],
  "suggestions": [
    {{"action": "建议操作", "reason": "原因说明"}}
  ]
}}
"""


class PromptOptimizer:
    """提示优化器"""
    
    def __init__(self):
        self.performance_history = []
    
    def optimize_prompt_for_content(self, content: ContentInput, base_prompt: str) -> str:
        """根据内容特征优化提示
        
        Args:
            content: 内容输入
            base_prompt: 基础提示
            
        Returns:
            str: 优化后的提示
        """
        optimizations = []
        
        # 根据文本长度调整
        combined_text = content.get_combined_text()
        if len(combined_text) > 1000:
            optimizations.append("注意：文本较长，请重点关注关键信息，避免被无关内容干扰。")
        
        # 根据来源类型调整
        sources = []
        if content.title: sources.append("标题")
        if content.body: sources.append("正文")
        if content.ocr_text: sources.append("OCR")
        if content.asr_text: sources.append("ASR")
        
        if len(sources) > 2:
            optimizations.append(f"内容包含多个来源({', '.join(sources)})，请综合分析并标注证据来源。")
        
        # 根据内容特征调整
        if "测评" in combined_text or "对比" in combined_text:
            optimizations.append("检测到测评/对比内容，请特别关注产品类目、功效和情感立场标签。")
        
        if "教程" in combined_text or "步骤" in combined_text:
            optimizations.append("检测到教程内容，请关注使用场景和产品类目标签。")
        
        if optimizations:
            optimization_text = "\n".join([f"- {opt}" for opt in optimizations])
            return f"{base_prompt}\n\n特别提示：\n{optimization_text}"
        
        return base_prompt
    
    def record_performance(self, content_id: str, extraction_time: float, 
                          quality_score: float, token_usage: int):
        """记录性能数据
        
        Args:
            content_id: 内容ID
            extraction_time: 抽取时间
            quality_score: 质量评分
            token_usage: token使用量
        """
        self.performance_history.append({
            'content_id': content_id,
            'extraction_time': extraction_time,
            'quality_score': quality_score,
            'token_usage': token_usage
        })
    
    def get_performance_stats(self) -> Dict[str, float]:
        """获取性能统计
        
        Returns:
            Dict: 性能统计数据
        """
        if not self.performance_history:
            return {}
        
        times = [p['extraction_time'] for p in self.performance_history]
        scores = [p['quality_score'] for p in self.performance_history]
        tokens = [p['token_usage'] for p in self.performance_history]
        
        return {
            'avg_extraction_time': sum(times) / len(times),
            'avg_quality_score': sum(scores) / len(scores),
            'avg_token_usage': sum(tokens) / len(tokens),
            'total_processed': len(self.performance_history)
        }


def create_extraction_prompt(content: ContentInput, taxonomy_path: str = None) -> str:
    """便捷函数：创建抽取提示
    
    Args:
        content: 内容输入
        taxonomy_path: 分类体系文件路径
        
    Returns:
        str: 完整的提示文本
    """
    template = PromptTemplate(taxonomy_path)
    system_prompt = template.get_system_prompt()
    user_prompt = template.get_user_prompt(content)
    
    return f"{system_prompt}\n\n{user_prompt}"