"""预处理模块

负责文本清洗、分段、来源标注等预处理任务。
"""

import re
import html
from typing import Dict, List, Optional, Tuple
from loguru import logger

from .schemas import ContentInput, SourceType
from .utils import clean_text, load_config


class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化预处理器
        
        Args:
            config: 配置字典
        """
        self.config = config or load_config()
        
        # 预处理配置
        pipeline_config = self.config.get('pipeline', {})
        self.max_chars_per_segment = pipeline_config.get('max_chars_per_segment', 1500)
        self.source_priority = pipeline_config.get('source_priority', ['body', 'title', 'asr', 'ocr'])
        
        # 编译正则表达式
        self._compile_patterns()
        
        logger.info("文本预处理器初始化完成")
    
    def _compile_patterns(self):
        """编译常用正则表达式"""
        # HTML标签
        self.html_pattern = re.compile(r'<[^>]+>')
        
        # 多余空白
        self.whitespace_pattern = re.compile(r'\s+')
        
        # 表情符号（保留常见的推荐表情）
        self.emoji_keep_pattern = re.compile(r'[👍👎💕❤️😍🥰😊😭😂🔥💯✨]')
        # 修复表情符号移除正则，避免误删中文字符
        self.emoji_remove_pattern = re.compile(
            r'[\U0001F600-\U0001F64F]|'  # 表情符号
            r'[\U0001F300-\U0001F5FF]|'  # 杂项符号和象形文字
            r'[\U0001F680-\U0001F6FF]|'  # 交通和地图符号
            r'[\U0001F1E0-\U0001F1FF]|'  # 区域指示符号
            r'[\U00002702-\U000027B0]|'  # 杂项符号
            r'[\U000024C2-\U000024FF]|'  # 封闭字母数字（修正范围）
            r'[\U0001F100-\U0001F1FF]|'  # 封闭字母数字补充
            r'[\U0001F200-\U0001F2FF]'   # 封闭CJK字母和月份
        )
        
        # URL链接
        self.url_pattern = re.compile(r'https?://[^\s]+|www\.[^\s]+')
        
        # @用户名
        self.mention_pattern = re.compile(r'@[\w\u4e00-\u9fff]+')
        
        # #话题标签
        self.hashtag_pattern = re.compile(r'#[\w\u4e00-\u9fff]+')
        
        # 重复标点
        self.repeat_punct_pattern = re.compile(r'([！？。，；：]){2,}')
        
        # 特殊字符清理
        self.special_chars_pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')
    
    def preprocess_content(self, content: ContentInput) -> ContentInput:
        """预处理内容
        
        Args:
            content: 原始内容
            
        Returns:
            ContentInput: 预处理后的内容
        """
        logger.debug(f"开始预处理内容: {content.content_id}")
        
        # 预处理各个字段
        processed_title = self._preprocess_text(content.title, 'title') if content.title else None
        processed_body = self._preprocess_text(content.body, 'body') if content.body else None
        processed_ocr = self._preprocess_text(content.ocr_text, 'ocr') if content.ocr_text else None
        processed_asr = self._preprocess_text(content.asr_text, 'asr') if content.asr_text else None
        
        # 应用长度限制
        if processed_body and len(processed_body) > 2000:
            logger.warning(f"内容过长被截断 {content.content_id}: {len(processed_body)} -> 2000")
            processed_body = processed_body[:2000]
        
        # 创建预处理后的内容
        processed_content = ContentInput(
            content_id=content.content_id,
            title=processed_title,
            body=processed_body,
            ocr_text=processed_ocr,
            asr_text=processed_asr,
            extra_fields=content.extra_fields
        )
        
        # 验证处理结果
        self._validate_processed_content(processed_content)
        
        logger.debug(f"完成预处理内容: {content.content_id}")
        return processed_content
    
    def _preprocess_text(self, text: str, source_type: str) -> str:
        """预处理单个文本
        
        Args:
            text: 原始文本
            source_type: 来源类型
            
        Returns:
            str: 预处理后的文本
        """
        if not text or not text.strip():
            return ""
        
        # 1. HTML解码
        text = html.unescape(text)
        
        # 2. 移除HTML标签
        text = self.html_pattern.sub('', text)
        
        # 3. 移除特殊控制字符
        text = self.special_chars_pattern.sub('', text)
        
        # 4. 处理URL和@用户名（根据来源类型决定是否保留）
        if source_type in ['title', 'body']:
            # 正文中的链接替换为占位符
            text = self.url_pattern.sub('[链接]', text)
            text = self.mention_pattern.sub('[用户]', text)
        else:
            # OCR/ASR中直接移除
            text = self.url_pattern.sub('', text)
            text = self.mention_pattern.sub('', text)
        
        # 5. 处理话题标签（保留但简化）
        text = self.hashtag_pattern.sub(lambda m: m.group(0)[1:], text)  # 移除#号
        
        # 6. 处理表情符号
        text = self._process_emojis(text)
        
        # 7. 标准化标点符号
        text = self._normalize_punctuation(text)
        
        # 8. 清理多余空白
        text = self.whitespace_pattern.sub(' ', text)
        text = text.strip()
        
        return text
    
    def _process_emojis(self, text: str) -> str:
        """处理表情符号
        
        Args:
            text: 输入文本
            
        Returns:
            str: 处理后的文本
        """
        # 保留重要的表情符号
        important_emojis = self.emoji_keep_pattern.findall(text)
        
        # 移除其他表情符号
        text = self.emoji_remove_pattern.sub('', text)
        
        # 将重要表情符号转换为文本描述
        emoji_map = {
            '👍': '[赞]',
            '👎': '[踩]', 
            '💕': '[爱心]',
            '❤️': '[红心]',
            '😍': '[花痴]',
            '🥰': '[可爱]',
            '😊': '[微笑]',
            '😭': '[哭]',
            '😂': '[笑哭]',
            '🔥': '[火]',
            '💯': '[满分]',
            '✨': '[闪亮]'
        }
        
        for emoji, desc in emoji_map.items():
            text = text.replace(emoji, desc)
        
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """标准化标点符号
        
        Args:
            text: 输入文本
            
        Returns:
            str: 标准化后的文本
        """
        # 统一标点符号
        punctuation_map = {
            '！！+': '！',
            '？？+': '？',
            '。。+': '。',
            '，，+': '，',
            '；；+': '；',
            '：：+': '：',
            '~~~+': '~',
            '---+': '-'
        }
        
        for pattern, replacement in punctuation_map.items():
            text = re.sub(pattern, replacement, text)
        
        # 处理重复标点
        text = self.repeat_punct_pattern.sub(r'\1', text)
        
        # 标准化引号
        text = re.sub(r'[""'']', '"', text)
        
        return text
    
    def _validate_processed_content(self, content: ContentInput):
        """验证预处理后的内容
        
        Args:
            content: 预处理后的内容
        """
        # 检查是否有有效内容
        has_content = any([
            content.title and content.title.strip(),
            content.body and content.body.strip(),
            content.ocr_text and content.ocr_text.strip(),
            content.asr_text and content.asr_text.strip()
        ])
        
        if not has_content:
            logger.warning(f"预处理后内容为空: {content.content_id}")
    
    def get_primary_text(self, content: ContentInput) -> Tuple[str, SourceType]:
        """获取主要文本内容
        
        Args:
            content: 内容对象
            
        Returns:
            Tuple[str, SourceType]: (主要文本, 来源类型)
        """
        # 按优先级选择主要文本
        for source in self.source_priority:
            if source == 'title' and content.title:
                return content.title, SourceType.TITLE
            elif source == 'body' and content.body:
                return content.body, SourceType.BODY
            elif source == 'asr' and content.asr_text:
                return content.asr_text, SourceType.ASR
            elif source == 'ocr' and content.ocr_text:
                return content.ocr_text, SourceType.OCR
        
        # 如果都没有，返回空字符串
        return "", SourceType.BODY
    
    def segment_text(self, text: str, max_chars: Optional[int] = None) -> List[str]:
        """分段文本
        
        Args:
            text: 输入文本
            max_chars: 最大字符数
            
        Returns:
            List[str]: 分段后的文本列表
        """
        if not text:
            return []
        
        max_chars = max_chars or self.max_chars_per_segment
        
        # 如果文本长度小于限制，直接返回
        if len(text) <= max_chars:
            return [text]
        
        segments = []
        current_pos = 0
        
        while current_pos < len(text):
            # 计算当前段的结束位置
            end_pos = current_pos + max_chars
            
            if end_pos >= len(text):
                # 最后一段
                segments.append(text[current_pos:])
                break
            
            # 寻找合适的分割点（句号、感叹号、问号）
            segment_text = text[current_pos:end_pos]
            
            # 从后往前找标点符号
            split_pos = -1
            for i in range(len(segment_text) - 1, -1, -1):
                if segment_text[i] in '。！？':
                    split_pos = i + 1
                    break
            
            if split_pos > 0 and split_pos > len(segment_text) * 0.5:
                # 找到合适的分割点
                segments.append(text[current_pos:current_pos + split_pos])
                current_pos += split_pos
            else:
                # 没找到合适的分割点，强制分割
                segments.append(text[current_pos:end_pos])
                current_pos = end_pos
        
        return segments


class PreprocessingPipeline:
    """预处理流水线"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else {}
        self.preprocessor = TextPreprocessor(self.config)
        
    def process(self, contents: List[ContentInput]) -> List[ContentInput]:
        """处理预处理"""
        logger.info(f"开始批量预处理，共 {len(contents)} 条内容")
        
        processed_contents = []
        
        for content in contents:
            try:
                processed_content = self.preprocessor.preprocess_content(content)
                processed_contents.append(processed_content)
                
            except Exception as e:
                logger.error(f"预处理失败 {content.content_id}: {e}")
                # 保留原始内容
                processed_contents.append(content)
                
        logger.info(f"批量预处理完成，成功处理 {len(processed_contents)} 条")
        return processed_contents


# 便捷函数
def create_preprocessing_pipeline(config_path: Optional[str] = None) -> PreprocessingPipeline:
    """创建预处理流水线"""
    return PreprocessingPipeline(config_path)


def preprocess_contents(
    contents: List[ContentInput],
    config_path: Optional[str] = None
) -> List[ContentInput]:
    """预处理内容列表"""
    pipeline = create_preprocessing_pipeline(config_path)
    return pipeline.process(contents)