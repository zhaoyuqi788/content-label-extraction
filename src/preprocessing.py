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
        self.emoji_remove_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+')
        
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
        
        # 9. 繁体转简体（如果需要）
        if self.config.get('preprocessing', {}).get('convert_traditional', False):
            text = self._convert_traditional_to_simplified(text)
        
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
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r'[''']', "'", text)
        
        return text
    
    def _convert_traditional_to_simplified(self, text: str) -> str:
        """繁体转简体（简单实现）
        
        Args:
            text: 输入文本
            
        Returns:
            str: 转换后的文本
        """
        # 这里可以集成更完整的繁简转换库，如 opencc
        # 目前提供基础的字符映射
        traditional_map = {
            '護': '护', '膚': '肤', '產': '产', '品': '品',
            '質': '质', '量': '量', '價': '价', '格': '格',
            '購': '购', '買': '买', '評': '评', '價': '价',
            '優': '优', '點': '点', '缺': '缺', '點': '点',
            '適': '适', '合': '合', '敏': '敏', '感': '感',
            '乾': '干', '燥': '燥', '油': '油', '性': '性',
            '混': '混', '合': '合', '性': '性', '肌': '肌',
            '膚': '肤', '質': '质', '問': '问', '題': '题',
            '效': '效', '果': '果', '成': '成', '分': '分',
            '濃': '浓', '度': '度', '質': '质', '地': '地',
            '顏': '颜', '色': '色', '味': '味', '道': '道'
        }
        
        for trad, simp in traditional_map.items():
            text = text.replace(trad, simp)
        
        return text
    
    def _validate_processed_content(self, content: ContentInput):
        """验证预处理后的内容
        
        Args:
            content: 预处理后的内容
            
        Raises:
            ValueError: 如果内容无效
        """
        # 检查是否有有效文本
        combined_text = content.get_combined_text()
        if not combined_text.strip():
            raise ValueError(f"预处理后没有有效文本内容: {content.content_id}")
        
        # 检查文本长度
        if len(combined_text) < 10:
            logger.warning(f"预处理后文本过短: {content.content_id}, 长度: {len(combined_text)}")
        
        # 检查字符编码
        try:
            combined_text.encode('utf-8')
        except UnicodeEncodeError as e:
            raise ValueError(f"文本编码错误: {content.content_id}, {e}")
    
    def get_source_priority_mapping(self) -> Dict[str, int]:
        """获取来源优先级映射
        
        Returns:
            Dict[str, int]: 来源到优先级的映射
        """
        return {source: i for i, source in enumerate(self.source_priority)}


class PreprocessingPipeline:
    """预处理流水线"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化预处理流水线
        
        Args:
            config: 配置字典
        """
        self.config = config or load_config()
        
        # 初始化组件
        self.text_preprocessor = TextPreprocessor(self.config)
        
        logger.info("预处理流水线初始化完成")
    
    def process(self, contents: List[ContentInput]) -> List[ContentInput]:
        """处理内容列表
        
        Args:
            contents: 输入内容列表
            
        Returns:
            List[ContentInput]: 预处理后的内容列表
        """
        logger.info(f"开始预处理 {len(contents)} 个内容")
        
        processed_contents = []
        
        for content in contents:
            try:
                # 文本预处理
                preprocessed_content = self.text_preprocessor.preprocess_content(content)
                processed_contents.append(preprocessed_content)
                
            except Exception as e:
                logger.error(f"预处理失败 {content.content_id}: {e}")
                # 可以选择跳过或使用原始内容
                continue
        
        logger.info(f"预处理完成: {len(contents)} -> {len(processed_contents)} 个内容")
        return processed_contents
    
    def process_single(self, content: ContentInput) -> ContentInput:
        """处理单个内容
        
        Args:
            content: 输入内容
            
        Returns:
            ContentInput: 预处理后的内容
        """
        return self.text_preprocessor.preprocess_content(content)


def create_preprocessing_pipeline(config_path: Optional[str] = None) -> PreprocessingPipeline:
    """便捷函数：创建预处理流水线
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        PreprocessingPipeline: 预处理流水线实例
    """
    config = load_config(config_path) if config_path else None
    return PreprocessingPipeline(config)


def preprocess_contents(contents: List[ContentInput], 
                       config_path: Optional[str] = None) -> List[ContentInput]:
    """便捷函数：预处理内容列表
    
    Args:
        contents: 内容列表
        config_path: 配置文件路径
        
    Returns:
        List[ContentInput]: 预处理后的内容列表
    """
    pipeline = create_preprocessing_pipeline(config_path)
    return pipeline.process(contents);
