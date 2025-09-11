"""数据模式定义

定义输入输出数据的Pydantic模型，确保数据结构的一致性和验证。
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class SourceType(str, Enum):
    """文本来源类型"""
    TITLE = "title"
    BODY = "body"
    OCR = "ocr"
    ASR = "asr"


class Evidence(BaseModel):
    """证据信息"""
    text: str = Field(..., max_length=50, description="证据文本，最多50字符")
    source: SourceType = Field(..., description="证据来源")


class LabelItem(BaseModel):
    """单个标签项"""
    label: str = Field(..., description="标签名称")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度，0-1之间")
    evidence: Evidence = Field(..., description="证据信息")


class BrandItem(BaseModel):
    """品牌标签项"""
    raw: str = Field(..., description="原始品牌名称")
    norm_id: Optional[str] = Field(None, description="标准化品牌ID")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    evidence: Evidence = Field(..., description="证据信息")


class StanceItem(BaseModel):
    """情感立场项"""
    label: str = Field(..., description="立场标签")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    evidence: Evidence = Field(..., description="证据信息")


class ContentInput(BaseModel):
    """输入内容模型"""
    content_id: str = Field(..., description="内容唯一ID")
    title: Optional[str] = Field(None, description="标题")
    body: Optional[str] = Field(None, description="正文内容")
    ocr_text: Optional[str] = Field(None, description="OCR文本")
    asr_text: Optional[str] = Field(None, description="ASR文本")
    extra_fields: Optional[Dict[str, Any]] = Field(default_factory=dict, description="额外字段")

    @validator('content_id')
    def validate_content_id(cls, v):
        if not v or not v.strip():
            raise ValueError('content_id不能为空')
        return v.strip()

    def get_combined_text(self) -> str:
        """获取合并后的文本"""
        texts = []
        if self.title:
            texts.append(f"标题: {self.title}")
        if self.body:
            texts.append(f"正文: {self.body}")
        if self.ocr_text:
            texts.append(f"OCR: {self.ocr_text}")
        if self.asr_text:
            texts.append(f"ASR: {self.asr_text}")
        return "\n\n".join(texts)

    def get_text_by_source(self, source: SourceType) -> Optional[str]:
        """根据来源获取文本"""
        mapping = {
            SourceType.TITLE: self.title,
            SourceType.BODY: self.body,
            SourceType.OCR: self.ocr_text,
            SourceType.ASR: self.asr_text
        }
        return mapping.get(source)


class ContentLabels(BaseModel):
    """内容标签输出模型"""
    content_id: str = Field(..., description="内容ID")
    talking_angles: List[LabelItem] = Field(default_factory=list, description="谈论角度")
    scenarios: List[LabelItem] = Field(default_factory=list, description="使用场景")
    skin_types: List[LabelItem] = Field(default_factory=list, description="肤质类型")
    skin_concerns: List[LabelItem] = Field(default_factory=list, description="肤况诉求")
    product_categories: List[LabelItem] = Field(default_factory=list, description="产品类目")
    ingredients: List[LabelItem] = Field(default_factory=list, description="成分")
    benefits: List[LabelItem] = Field(default_factory=list, description="功效")
    brands: List[BrandItem] = Field(default_factory=list, description="品牌")
    stance: Optional[StanceItem] = Field(None, description="情感立场")
    compliance_flags: List[LabelItem] = Field(default_factory=list, description="合规标识")
    quality_flags: List[LabelItem] = Field(default_factory=list, description="质量标识")
    notes: Optional[str] = Field(None, description="备注说明")

    def get_all_labels(self) -> Dict[str, List[Union[LabelItem, BrandItem, StanceItem]]]:
        """获取所有标签"""
        result = {
            'talking_angles': self.talking_angles,
            'scenarios': self.scenarios,
            'skin_types': self.skin_types,
            'skin_concerns': self.skin_concerns,
            'product_categories': self.product_categories,
            'ingredients': self.ingredients,
            'benefits': self.benefits,
            'brands': self.brands,
            'compliance_flags': self.compliance_flags,
            'quality_flags': self.quality_flags
        }
        if self.stance:
            result['stance'] = [self.stance]
        return result


class ProcessingResult(BaseModel):
    """处理结果模型"""
    content_id: str = Field(..., description="内容ID")
    success: bool = Field(..., description="处理是否成功")
    labels: Optional[ContentLabels] = Field(None, description="提取的标签")
    error_message: Optional[str] = Field(None, description="错误信息")
    processing_time: Optional[float] = Field(None, description="处理时间（秒）")
    tokens_used: Optional[int] = Field(None, description="使用的token数量")


class BatchProcessingResult(BaseModel):
    """批处理结果模型"""
    total_count: int = Field(..., description="总数量")
    success_count: int = Field(..., description="成功数量")
    failed_count: int = Field(..., description="失败数量")
    results: List[ProcessingResult] = Field(..., description="处理结果列表")
    total_processing_time: float = Field(..., description="总处理时间")
    total_tokens_used: int = Field(default=0, description="总token使用量")

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count


class RuleMatch(BaseModel):
    """规则匹配结果"""
    rule_name: str = Field(..., description="规则名称")
    matched_text: str = Field(..., description="匹配的文本")
    label: str = Field(..., description="标签")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度")
    source: SourceType = Field(..., description="来源")
    category: str = Field(..., description="标签类别")


class LLMExtractionRequest(BaseModel):
    """LLM抽取请求"""
    content_id: str = Field(..., description="内容ID")
    text: str = Field(..., description="待抽取文本")
    source_mapping: Dict[str, SourceType] = Field(..., description="文本片段到来源的映射")
    existing_labels: Optional[Dict[str, List[LabelItem]]] = Field(None, description="已有标签")


class LLMExtractionResponse(BaseModel):
    """LLM抽取响应"""
    content_id: str = Field(..., description="内容ID")
    extracted_labels: ContentLabels = Field(..., description="提取的标签")
    tokens_used: int = Field(default=0, description="使用的token数量")
    processing_time: float = Field(default=0.0, description="处理时间")
    model_name: str = Field(..., description="使用的模型名称")