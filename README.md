# 美妆内容标签抽取系统

一个基于规则引擎和大语言模型的智能美妆内容标签抽取系统，能够从美妆相关文本中自动识别和提取品牌、成分、功效等关键信息。

## 功能特性

- **多模态抽取**: 结合规则引擎和LLM的混合抽取策略
- **高精度识别**: 基于美妆领域知识库的精准匹配
- **批量处理**: 支持大规模内容的批量标签抽取
- **结果融合**: 智能融合多种抽取方法的结果
- **标准化输出**: 自动归一化品牌和成分名称
- **多格式导出**: 支持JSON、CSV、Parquet等格式

## 系统架构

```
输入内容 → 预处理 → 规则引擎抽取 → LLM抽取 → 结果融合 → 归一化 → 导出
```

### 核心模块

1. **预处理模块** (`preprocessing.py`): 文本清洗、分段、标准化
2. **规则引擎** (`rules.py`): 基于词典和正则的高置信度抽取
3. **LLM客户端** (`llm_client.py`): 支持多种大语言模型API
4. **标签抽取器** (`extractor.py`): 统一的抽取接口
5. **结果融合** (`fusion.py`): 多源结果的智能融合
6. **归一化器** (`normalizer.py`): 标签标准化和别名处理
7. **导出器** (`exporter.py`): 多格式结果导出

## 安装说明

### 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 安装步骤

1. 克隆项目
```bash
git clone <repository-url>
cd content-label-extraction
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

## 配置说明

### 1. LLM配置

在 `config/llm_config.yaml` 中配置LLM服务:

```yaml
llm_providers:
  openai:
    api_key: "your-openai-api-key"
    base_url: "https://api.openai.com/v1"
    model: "gpt-3.5-turbo"
  
  dashscope:
    api_key: "your-dashscope-api-key"
    model: "qwen-turbo"
```

### 2. 规则配置

在 `config/rules_config.yaml` 中配置抽取规则:

```yaml
brands:
  - "兰蔻"
  - "雅诗兰黛"
  - "SK-II"

ingredients:
  - "透明质酸"
  - "烟酰胺"
  - "维生素C"
```

## 使用方法

### 批量处理

```bash
python src/run_batch.py \
  --input data/input.jsonl \
  --output data/output.jsonl \
  --config config/default_config.yaml \
  --format jsonl
```

### 参数说明

- `--input`: 输入文件路径（支持JSONL、CSV、Excel）
- `--output`: 输出文件路径
- `--config`: 配置文件路径
- `--format`: 输出格式（jsonl/csv/parquet）
- `--batch-size`: 批处理大小（默认100）
- `--workers`: 并发工作线程数（默认4）

### 编程接口

```python
from src.run_batch import BatchLabelingPipeline

# 创建流水线
pipeline = BatchLabelingPipeline(
    config_path="config/default_config.yaml"
)

# 处理内容
contents = [
    {"id": "1", "text": "兰蔻小黑瓶含有透明质酸成分", "source": "product_desc"},
    {"id": "2", "text": "雅诗兰黛红石榴系列适合抗氧化", "source": "review"}
]

results = pipeline.process_batch(contents)
print(results)
```

## 输入格式

### JSONL格式
```json
{"id": "1", "text": "产品描述文本", "source": "product_desc"}
{"id": "2", "text": "用户评价文本", "source": "review"}
```

### CSV格式
```csv
id,text,source
1,"产品描述文本",product_desc
2,"用户评价文本",review
```

## 输出格式

```json
{
  "id": "1",
  "original_text": "兰蔻小黑瓶含有透明质酸成分",
  "source": "product_desc",
  "extracted_labels": {
    "brands": [
      {"name": "兰蔻", "confidence": 0.95, "method": "rule"}
    ],
    "ingredients": [
      {"name": "透明质酸", "confidence": 0.90, "method": "rule"}
    ],
    "effects": [
      {"name": "保湿", "confidence": 0.85, "method": "llm"}
    ]
  },
  "processing_time": 0.234,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 性能优化

1. **批处理**: 使用适当的批处理大小（推荐100-500）
2. **并发处理**: 根据硬件配置调整工作线程数
3. **缓存机制**: LLM结果自动缓存，避免重复调用
4. **规则优先**: 高置信度规则结果优先，减少LLM调用

## 扩展开发

### 添加新的LLM提供商

1. 在 `llm_client.py` 中继承 `BaseLLMClient`
2. 实现 `_make_request` 方法
3. 在配置文件中添加相应配置

### 添加新的抽取规则

1. 在 `rules.py` 中继承 `BaseMatcher`
2. 实现 `match` 方法
3. 在 `RuleEngine` 中注册新规则

### 自定义导出格式

1. 在 `exporter.py` 中继承 `BaseExporter`
2. 实现 `export` 方法
3. 在 `LabelExporter` 中注册新格式

## 故障排除

### 常见问题

1. **LLM API调用失败**
   - 检查API密钥配置
   - 确认网络连接
   - 查看API配额限制

2. **内存不足**
   - 减少批处理大小
   - 降低并发线程数
   - 使用流式处理

3. **抽取精度不高**
   - 优化规则配置
   - 调整LLM提示词
   - 增加训练数据

### 日志查看

系统使用loguru进行日志记录，日志文件位于 `logs/` 目录下。

```bash
# 查看最新日志
tail -f logs/extraction.log

# 查看错误日志
grep ERROR logs/extraction.log
```

## 许可证

MIT License

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request