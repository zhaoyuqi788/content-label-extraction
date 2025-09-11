"""工具函数模块

提供配置加载、文本处理、日志设置等通用功能。
"""

import yaml
import re
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger
import sys


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径，默认为config/config.yaml
        
    Returns:
        Dict: 配置字典
    """
    if config_path is None:
        # 获取项目根目录
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        config_path = project_root / "config" / "config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        raise


def load_taxonomy(taxonomy_path: Optional[str] = None) -> Dict[str, List[str]]:
    """加载标签分类体系
    
    Args:
        taxonomy_path: 分类体系文件路径，默认为config/taxonomy.yaml
        
    Returns:
        Dict: 标签分类体系
    """
    if taxonomy_path is None:
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        taxonomy_path = project_root / "config" / "taxonomy.yaml"
    
    taxonomy_path = Path(taxonomy_path)
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"分类体系文件不存在: {taxonomy_path}")
    
    try:
        with open(taxonomy_path, 'r', encoding='utf-8') as f:
            taxonomy = yaml.safe_load(f)
        logger.info(f"成功加载标签分类体系: {taxonomy_path}")
        return taxonomy
    except Exception as e:
        logger.error(f"加载标签分类体系失败: {e}")
        raise


def setup_logging(config: Dict[str, Any]) -> None:
    """设置日志配置
    
    Args:
        config: 配置字典
    """
    log_config = config.get('logging', {})
    
    # 移除默认处理器
    logger.remove()
    
    # 添加控制台处理器
    logger.add(
        sys.stderr,
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}'),
        colorize=True
    )
    
    # 添加文件处理器
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger.add(
        log_dir / "content_labeling.log",
        level=log_config.get('level', 'INFO'),
        format=log_config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}'),
        rotation=log_config.get('rotation', '10 MB'),
        retention=log_config.get('retention', '7 days'),
        encoding='utf-8'
    )
    
    logger.info("日志系统初始化完成")


def clean_text(text: str) -> str:
    """清洗文本
    
    Args:
        text: 原始文本
        
    Returns:
        str: 清洗后的文本
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 去除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 统一换行符
    text = re.sub(r'\r\n|\r', '\n', text)
    
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空白
    text = text.strip()
    
    return text


def truncate_text(text: str, max_length: int = 2000) -> str:
    """截断文本
    
    Args:
        text: 原始文本
        max_length: 最大长度
        
    Returns:
        str: 截断后的文本
    """
    if not text or len(text) <= max_length:
        return text
    
    # 在句号、感叹号、问号处截断
    truncated = text[:max_length]
    
    # 寻找最后一个句子结束符
    last_sentence_end = max(
        truncated.rfind('。'),
        truncated.rfind('！'),
        truncated.rfind('？'),
        truncated.rfind('.'),
        truncated.rfind('!'),
        truncated.rfind('?')
    )
    
    if last_sentence_end > max_length * 0.8:  # 如果截断点在80%之后
        return truncated[:last_sentence_end + 1]
    else:
        return truncated


def extract_evidence(text: str, keyword: str, max_length: int = 50) -> str:
    """提取证据文本
    
    Args:
        text: 原始文本
        keyword: 关键词
        max_length: 最大长度
        
    Returns:
        str: 证据文本
    """
    if not text or not keyword:
        return ""
    
    # 查找关键词位置
    keyword_pos = text.lower().find(keyword.lower())
    if keyword_pos == -1:
        return text[:max_length] if len(text) > max_length else text
    
    # 计算前后文长度
    context_length = (max_length - len(keyword)) // 2
    
    start = max(0, keyword_pos - context_length)
    end = min(len(text), keyword_pos + len(keyword) + context_length)
    
    evidence = text[start:end]
    
    # 添加省略号
    if start > 0:
        evidence = "..." + evidence
    if end < len(text):
        evidence = evidence + "..."
    
    return evidence


def get_api_key(key_name: str) -> str:
    """获取API密钥
    
    Args:
        key_name: 环境变量名称
        
    Returns:
        str: API密钥
        
    Raises:
        ValueError: 密钥不存在
    """
    api_key = os.getenv(key_name)
    if not api_key:
        raise ValueError(f"环境变量 {key_name} 未设置")
    return api_key


def calculate_confidence_fusion(confidences: List[float], method: str = "weighted_max") -> float:
    """计算置信度融合
    
    Args:
        confidences: 置信度列表
        method: 融合方法 (weighted_max, average, max)
        
    Returns:
        float: 融合后的置信度
    """
    if not confidences:
        return 0.0
    
    if method == "max":
        return max(confidences)
    elif method == "average":
        return sum(confidences) / len(confidences)
    elif method == "weighted_max":
        # 加权最大值：最高置信度 + 其他置信度的加权平均
        max_conf = max(confidences)
        if len(confidences) == 1:
            return max_conf
        
        other_confs = [c for c in confidences if c != max_conf]
        avg_other = sum(other_confs) / len(other_confs)
        
        # 权重：最高置信度占70%，其他占30%
        return min(1.0, max_conf * 0.7 + avg_other * 0.3)
    else:
        raise ValueError(f"不支持的融合方法: {method}")


def is_chinese_text(text: str) -> bool:
    """判断是否为中文文本
    
    Args:
        text: 待判断文本
        
    Returns:
        bool: 是否为中文文本
    """
    if not text:
        return False
    
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    return len(chinese_chars) / len(text) > 0.3


def normalize_brand_name(brand_name: str) -> str:
    """标准化品牌名称
    
    Args:
        brand_name: 原始品牌名称
        
    Returns:
        str: 标准化后的品牌名称
    """
    if not brand_name:
        return ""
    
    # 去除特殊字符和空格
    normalized = re.sub(r'[^\w\u4e00-\u9fff]', '', brand_name.lower())
    
    return normalized


def create_output_dir(output_path: str) -> Path:
    """创建输出目录
    
    Args:
        output_path: 输出路径
        
    Returns:
        Path: 输出路径对象
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def format_processing_time(seconds: float) -> str:
    """格式化处理时间
    
    Args:
        seconds: 秒数
        
    Returns:
        str: 格式化的时间字符串
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m{remaining_seconds:.1f}s"