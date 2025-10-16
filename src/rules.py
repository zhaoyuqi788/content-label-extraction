"""规则引擎模块

基于词典和正则表达式的高置信度标签抽取。
"""

import re
import csv
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from loguru import logger

from src.schemas import ContentInput, LabelItem, Evidence, SourceType
from src.utils import load_config, load_taxonomy


class RulePattern:
    """规则模式"""
    
    def __init__(self, pattern: str, label: str, confidence: float, 
                 category: str, description: str = ""):
        """初始化规则模式
        
        Args:
            pattern: 正则表达式模式
            label: 标签名称
            confidence: 置信度
            category: 标签类别
            description: 描述
        """
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.label = label
        self.confidence = confidence
        self.category = category
        self.description = description
    
    def match(self, text: str, source: SourceType) -> List[LabelItem]:
        """匹配文本
        
        Args:
            text: 输入文本
            source: 文本来源
            
        Returns:
            List[LabelItem]: 匹配的标签列表
        """
        matches = []
        for match in self.pattern.finditer(text):
            evidence_text = match.group(0)
            # 扩展上下文
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].strip()
            
            # 限制证据长度
            if len(context) > 50:
                context = context[:47] + "..."
            
            evidence = Evidence(text=context, source=source)
            
            label_item = LabelItem(
                label=self.label,
                confidence=self.confidence,
                evidence=evidence
            )
            matches.append(label_item)
        
        return matches


class KeywordMatcher:
    """关键词匹配器"""
    
    def __init__(self):
        """初始化关键词匹配器"""
        self.keyword_rules = {}
        self._build_keyword_patterns()
    
    def _build_keyword_patterns(self):
        """构建关键词模式"""
        # 谈论角度关键词
        talking_angle_keywords = {
            '种草': ['种草', '推荐', '安利', '必买', '回购', '无限回购', '强推', '墙裂推荐'],
            '拔草': ['拔草', '踩雷', '不推荐', '避雷', '慎买', '不值得', '后悔'],
            '测评': ['测评', '评测', '试用', '体验', '使用感', '真实感受'],
            '教程': ['教程', '步骤', '方法', '技巧', '怎么用', '如何使用', '使用方法'],
            '开箱': ['开箱', '拆箱', '新品', '首次', '第一次用'],
            '对比': ['对比', '比较', 'vs', 'PK', '哪个好', '区别'],
            '空瓶': ['空瓶', '用完了', '已空', '见底'],
            '试色': ['试色', '色号', '颜色', '色彩', '显色'],
            '科普': ['科普', '知识', '原理', '成分分析', '功效'],
            'Vlog': ['vlog', '日常', '一天', '生活'],
            '优惠信息': ['优惠', '折扣', '特价', '活动', '促销', '打折'],
            '心得': ['心得', '感受', '总结', '体会', '经验']
        }
        
        # 使用场景关键词
        scenario_keywords = {
            '通勤': ['通勤', '上班', '工作', '职场'],
            '约会': ['约会', '见面', '聚会', 'party'],
            '熬夜修护': ['熬夜', '修护', '熬夜后', '晚睡'],
            '健身': ['健身', '运动', '出汗', '锻炼'],
            '旅行': ['旅行', '出差', '度假', '旅游'],
            '婚礼': ['婚礼', '结婚', '新娘', '伴娘'],
            '换季': ['换季', '季节', '春夏', '秋冬'],
            '夏季控油': ['夏天', '控油', '出油', '油腻'],
            '冬季保湿': ['冬天', '保湿', '干燥', '补水'],
            '淡妆': ['淡妆', '日常妆', '素颜', '自然'],
            '浓妆': ['浓妆', '晚妆', '派对妆', '舞台妆'],
            '敏感期': ['敏感期', '生理期', '姨妈期', '经期'],
            '医美术后': ['医美', '术后', '激光后', '微整'],
            '户外防晒': ['户外', '防晒', '海边', '爬山']
        }
        
        # 肤质关键词
        skin_type_keywords = {
            '干性': ['干皮', '干性', '大干皮', '沙漠皮', '缺水'],
            '油性': ['油皮', '油性', '大油皮', '出油'],
            '混合': ['混合', '混油', '混干', 'T区油'],
            '中性': ['中性', '正常', '不油不干'],
            '敏感': ['敏感', '敏感肌', '易过敏', '红血丝'],
            '痘痘肌': ['痘痘肌', '长痘', '痘肌', '爆痘']
        }
        
        # 肤况关键词
        skin_concern_keywords = {
            '泛红': ['泛红', '发红', '红血丝', '潮红'],
            '闭口': ['闭口', '闭合性粉刺', '白头'],
            '黑头': ['黑头', '草莓鼻', '毛孔黑'],
            '毛孔': ['毛孔', '毛孔粗大', '毛孔明显'],
            '痘印': ['痘印', '痘疤', '痘坑', '印子'],
            '暗沉': ['暗沉', '发黄', '蜡黄', '无光泽'],
            '细纹': ['细纹', '皱纹', '法令纹', '眼纹'],
            '松弛': ['松弛', '下垂', '紧致', '提拉'],
            '出油': ['出油', '油腻', '油光', '泛油'],
            '脱皮': ['脱皮', '起皮', '掉皮屑'],
            '屏障受损': ['屏障', '受损', '破皮', '刺痛'],
            '水油不平衡': ['水油', '不平衡', '外油内干']
        }
        
        # 成分关键词
        ingredient_keywords = {
            '神经酰胺': ['神经酰胺', 'ceramide'],
            '玻尿酸': ['玻尿酸', '透明质酸', 'hyaluronic'],
            '烟酰胺': ['烟酰胺', 'niacinamide', '维生素B3'],
            '水杨酸': ['水杨酸', 'salicylic', 'BHA'],
            '壬二酸': ['壬二酸', 'azelaic'],
            'A醇': ['A醇', '视黄醇', 'retinol'],
            'VC': ['VC', '维C', '维生素C', 'vitamin c'],
            '角鲨烷': ['角鲨烷', 'squalane'],
            '积雪草': ['积雪草', 'centella']
        }
        
        # 功效关键词
        benefit_keywords = {
            '保湿': ['保湿', '补水', '滋润', '水润'],
            '修护': ['修护', '修复', '舒缓', '镇静'],
            '抗老': ['抗老', '抗衰', '紧致', '提拉'],
            '祛痘': ['祛痘', '抗痘', '消痘', '治痘'],
            '美白提亮': ['美白', '提亮', '亮白', '淡斑'],
            '控油': ['控油', '去油', '平衡油脂'],
            '舒缓': ['舒缓', '镇静', '消炎', '抗敏'],
            '防晒': ['防晒', 'SPF', 'PA', '防紫外线'],
            '清洁': ['清洁', '洁面', '去污', '深层清洁']
        }
        
        # 合规关键词
        compliance_keywords = {
            '可能广告/合作': ['广告', '合作', '赞助', '推广', '种草官', 'PR'],
            '夸大功效': ['根治', '永久', '立即见效', '一次见效', '神器'],
            '医疗化用语': ['治疗', '处方', '药用', '临床', '医学'],
            '敏感词': ['激素', '依赖', '成瘾', '副作用']
        }
        
        # 构建所有关键词规则
        all_keywords = {
            'talking_angles': talking_angle_keywords,
            'scenarios': scenario_keywords,
            'skin_types': skin_type_keywords,
            'skin_concerns': skin_concern_keywords,
            'ingredients': ingredient_keywords,
            'benefits': benefit_keywords,
            'compliance_flags': compliance_keywords
        }
        
        for category, keywords_dict in all_keywords.items():
            self.keyword_rules[category] = {}
            for label, keywords in keywords_dict.items():
                # 构建正则模式
                pattern = '|'.join(re.escape(kw) for kw in keywords)
                confidence = 0.9 if category == 'compliance_flags' else 0.85
                
                self.keyword_rules[category][label] = RulePattern(
                    pattern=f'({pattern})',
                    label=label,
                    confidence=confidence,
                    category=category
                )
    
    def match_keywords(self, text: str, source: SourceType) -> Dict[str, List[LabelItem]]:
        """匹配关键词
        
        Args:
            text: 输入文本
            source: 文本来源
            
        Returns:
            Dict[str, List[LabelItem]]: 按类别分组的标签列表
        """
        results = {}
        
        for category, rules in self.keyword_rules.items():
            category_matches = []
            
            for label, rule in rules.items():
                matches = rule.match(text, source)
                category_matches.extend(matches)
            
            if category_matches:
                results[category] = category_matches
        
        return results


class BrandMatcher:
    """品牌匹配器"""
    
    def __init__(self, config_dir: str):
        """初始化品牌匹配器
        
        Args:
            config_dir: 配置目录路径
        """
        self.brand_aliases = {}
        self.brand_patterns = {}
        self._load_brand_data(config_dir)
    
    def _load_brand_data(self, config_dir: str):
        """加载品牌数据
        
        Args:
            config_dir: 配置目录路径
        """
        brand_file = Path(config_dir) / 'synonyms_brand_aliases.csv'
        
        if not brand_file.exists():
            logger.warning(f"品牌别名文件不存在: {brand_file}")
            return
        
        try:
            with open(brand_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    brand_name = row['brand_name']
                    aliases = row['aliases'].split('|') if row['aliases'] else []
                    norm_id = row.get('norm_id', brand_name)
                    
                    # 存储别名映射
                    all_names = [brand_name] + aliases
                    for name in all_names:
                        self.brand_aliases[name.lower()] = {
                            'norm_id': norm_id,
                            'canonical_name': brand_name
                        }
                    
                    # 构建正则模式
                    pattern = '|'.join(re.escape(name) for name in all_names)
                    self.brand_patterns[brand_name] = re.compile(
                        f'({pattern})', re.IGNORECASE
                    )
            
            logger.info(f"加载品牌数据: {len(self.brand_patterns)} 个品牌")
            
        except Exception as e:
            logger.error(f"加载品牌数据失败: {e}")
    
    def match_brands(self, text: str, source: SourceType) -> List[LabelItem]:
        """匹配品牌
        
        Args:
            text: 输入文本
            source: 文本来源
            
        Returns:
            List[LabelItem]: 品牌标签列表
        """
        matches = []
        matched_brands = set()
        
        for brand_name, pattern in self.brand_patterns.items():
            for match in pattern.finditer(text):
                if brand_name in matched_brands:
                    continue
                
                matched_text = match.group(0)
                brand_info = self.brand_aliases.get(matched_text.lower())
                
                if brand_info:
                    # 扩展上下文
                    start = max(0, match.start() - 15)
                    end = min(len(text), match.end() + 15)
                    context = text[start:end].strip()
                    
                    if len(context) > 50:
                        context = context[:47] + "..."
                    
                    evidence = Evidence(text=context, source=source)
                    
                    # 创建品牌标签（特殊格式）
                    label_item = LabelItem(
                        label=brand_info['canonical_name'],
                        confidence=0.95,
                        evidence=evidence,
                        extra_data={
                            'raw': matched_text,
                            'norm_id': brand_info['norm_id']
                        }
                    )
                    
                    matches.append(label_item)
                    matched_brands.add(brand_name)
        
        return matches


class IngredientMatcher:
    """成分匹配器"""
    
    def __init__(self, config_dir: str):
        """初始化成分匹配器
        
        Args:
            config_dir: 配置目录路径
        """
        self.ingredient_data = {}
        self.ingredient_patterns = {}
        self._load_ingredient_data(config_dir)
    
    def _load_ingredient_data(self, config_dir: str):
        """加载成分数据
        
        Args:
            config_dir: 配置目录路径
        """
        ingredient_file = Path(config_dir) / 'ingredients_dict.csv'
        
        if not ingredient_file.exists():
            logger.warning(f"成分词典文件不存在: {ingredient_file}")
            return
        
        try:
            with open(ingredient_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ingredient_name = row['ingredient_name']
                    aliases = row['aliases'].split('|') if row['aliases'] else []
                    category = row.get('category', '')
                    benefits = row.get('benefits', '')
                    
                    # 存储成分数据
                    self.ingredient_data[ingredient_name] = {
                        'aliases': aliases,
                        'category': category,
                        'benefits': benefits
                    }
                    
                    # 构建正则模式
                    all_names = [ingredient_name] + aliases
                    pattern = '|'.join(re.escape(name) for name in all_names)
                    
                    # 添加浓度匹配
                    concentration_pattern = f'({pattern})\\s*[（(]?\\s*\\d+(?:\\.\\d+)?\\s*%\\s*[）)]?'
                    
                    self.ingredient_patterns[ingredient_name] = {
                        'basic': re.compile(f'({pattern})', re.IGNORECASE),
                        'with_concentration': re.compile(concentration_pattern, re.IGNORECASE)
                    }
            
            logger.info(f"加载成分数据: {len(self.ingredient_patterns)} 个成分")
            
        except Exception as e:
            logger.error(f"加载成分数据失败: {e}")
    
    def match_ingredients(self, text: str, source: SourceType) -> List[LabelItem]:
        """匹配成分
        
        Args:
            text: 输入文本
            source: 文本来源
            
        Returns:
            List[LabelItem]: 成分标签列表
        """
        matches = []
        matched_ingredients = set()
        
        for ingredient_name, patterns in self.ingredient_patterns.items():
            if ingredient_name in matched_ingredients:
                continue
            
            # 优先匹配带浓度的
            concentration_matches = list(patterns['with_concentration'].finditer(text))
            if concentration_matches:
                for match in concentration_matches:
                    context = self._extract_context(text, match)
                    evidence = Evidence(text=context, source=source)
                    
                    label_item = LabelItem(
                        label=ingredient_name,
                        confidence=0.95,  # 带浓度的置信度更高
                        evidence=evidence
                    )
                    
                    matches.append(label_item)
                    matched_ingredients.add(ingredient_name)
                    break
            else:
                # 匹配基础名称
                basic_matches = list(patterns['basic'].finditer(text))
                if basic_matches:
                    match = basic_matches[0]  # 只取第一个匹配
                    context = self._extract_context(text, match)
                    evidence = Evidence(text=context, source=source)
                    
                    label_item = LabelItem(
                        label=ingredient_name,
                        confidence=0.85,
                        evidence=evidence
                    )
                    
                    matches.append(label_item)
                    matched_ingredients.add(ingredient_name)
        
        return matches
    
    def _extract_context(self, text: str, match) -> str:
        """提取上下文
        
        Args:
            text: 原文本
            match: 匹配对象
            
        Returns:
            str: 上下文文本
        """
        start = max(0, match.start() - 20)
        end = min(len(text), match.end() + 20)
        context = text[start:end].strip()
        
        if len(context) > 50:
            context = context[:47] + "..."
        
        return context


class RuleEngine:
    """规则引擎"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化规则引擎
        
        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        
        # 获取配置目录
        if config_path:
            config_dir = Path(config_path).parent
        else:
            config_dir = Path('config')
        
        # 初始化匹配器
        self.keyword_matcher = KeywordMatcher()
        self.brand_matcher = BrandMatcher(str(config_dir))
        self.ingredient_matcher = IngredientMatcher(str(config_dir))
        
        # 规则配置
        rule_config = self.config.get('rules', {})
        self.min_confidence = rule_config.get('min_confidence', 0.8)
        self.max_matches_per_category = rule_config.get('max_matches_per_category', 10)
        
        logger.info("规则引擎初始化完成")
    
    def extract_labels(self, content: ContentInput) -> Dict[str, List[LabelItem]]:
        """提取标签
        
        Args:
            content: 输入内容
            
        Returns:
            Dict[str, List[LabelItem]]: 按类别分组的标签
        """
        logger.debug(f"规则引擎开始处理: {content.content_id}")
        
        all_labels = {}
        
        # 处理各个文本字段
        text_sources = [
            (content.title, SourceType.TITLE),
            (content.body, SourceType.BODY),
            (content.ocr_text, SourceType.OCR),
            (content.asr_text, SourceType.ASR)
        ]
        
        for text, source in text_sources:
            if not text:
                continue
            
            # 关键词匹配
            keyword_results = self.keyword_matcher.match_keywords(text, source)
            self._merge_results(all_labels, keyword_results)
            
            # 品牌匹配
            brand_matches = self.brand_matcher.match_brands(text, source)
            if brand_matches:
                if 'brands' not in all_labels:
                    all_labels['brands'] = []
                all_labels['brands'].extend(brand_matches)
            
            # 成分匹配
            ingredient_matches = self.ingredient_matcher.match_ingredients(text, source)
            if ingredient_matches:
                if 'ingredients' not in all_labels:
                    all_labels['ingredients'] = []
                all_labels['ingredients'].extend(ingredient_matches)
        
        # 后处理
        processed_labels = self._postprocess_labels(all_labels)
        
        logger.debug(f"规则引擎完成处理: {content.content_id}, "
                    f"提取标签: {sum(len(labels) for labels in processed_labels.values())}")
        
        return processed_labels
    
    def _merge_results(self, all_labels: Dict[str, List[LabelItem]], 
                      new_results: Dict[str, List[LabelItem]]):
        """合并结果
        
        Args:
            all_labels: 所有标签字典
            new_results: 新结果字典
        """
        for category, labels in new_results.items():
            if category not in all_labels:
                all_labels[category] = []
            all_labels[category].extend(labels)
    
    def _postprocess_labels(self, labels: Dict[str, List[LabelItem]]) -> Dict[str, List[LabelItem]]:
        """后处理标签
        
        Args:
            labels: 原始标签字典
            
        Returns:
            Dict[str, List[LabelItem]]: 处理后的标签字典
        """
        processed = {}
        
        for category, label_list in labels.items():
            # 过滤低置信度
            filtered_labels = [
                label for label in label_list 
                if label.confidence >= self.min_confidence
            ]
            
            # 去重（基于标签名称）
            seen_labels = set()
            unique_labels = []
            
            for label in filtered_labels:
                if label.label not in seen_labels:
                    unique_labels.append(label)
                    seen_labels.add(label.label)
            
            # 按置信度排序
            unique_labels.sort(key=lambda x: x.confidence, reverse=True)
            
            # 限制数量
            if len(unique_labels) > self.max_matches_per_category:
                unique_labels = unique_labels[:self.max_matches_per_category]
            
            if unique_labels:
                processed[category] = unique_labels
        
        return processed
    
    def get_rule_stats(self) -> Dict[str, Any]:
        """获取规则统计
        
        Returns:
            Dict[str, Any]: 规则统计信息
        """
        return {
            'keyword_categories': len(self.keyword_matcher.keyword_rules),
            'brand_count': len(self.brand_matcher.brand_patterns),
            'ingredient_count': len(self.ingredient_matcher.ingredient_patterns),
            'min_confidence': self.min_confidence,
            'max_matches_per_category': self.max_matches_per_category
        }


def create_rule_engine(config_path: Optional[str] = None) -> RuleEngine:
    """便捷函数：创建规则引擎
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        RuleEngine: 规则引擎实例
    """
    return RuleEngine(config_path)


def extract_labels_by_rules(content: ContentInput, 
                           config_path: Optional[str] = None) -> Dict[str, List[LabelItem]]:
    """便捷函数：使用规则提取标签
    
    Args:
        content: 输入内容
        config_path: 配置文件路径
        
    Returns:
        Dict[str, List[LabelItem]]: 提取的标签
    """
    engine = create_rule_engine(config_path)
    return engine.extract_labels(content)