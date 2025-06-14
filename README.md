# Math-Eval - 大模型数学能力高效评估框架

[English](README_EN.md) | 中文简体

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-0.4.1+-green.svg)](https://github.com/vllm-project/vllm)

Math-Eval是一个专门为大语言模型数学能力评估设计的高效评估框架。该框架集成了先进的并行推理引擎、双重答案验证机制和模块化设计，为研究人员和开发者提供了一个全面、可靠的数学能力评测工具。

## 🚀 核心优势

### 1. 高效并行推理引擎
基于vLLM异步调用引擎，支持张量并行（Tensor Parallelism）与数据并行（Data Parallelism）的混合加速策略。多benchmark评估时只需单次启动vLLM引擎，最大限度提升GPU利用率，显著降低评估时间成本。

### 2. 双重答案验证机制
- **规则验证**：集成Qwen-Math的答案提取方法与math_verify库的标准化数值验证流程，确保数值计算的准确性
- **模型验证**：基于xVerify框架支持本地模型或API服务的智能验证，提供更灵活的验证策略
- **支持自定义验证策略组合**：可配置AND/OR逻辑组合，满足不同评估需求

### 3. 模块化设计
- **可插拔式benchmark加载器**：支持JSON格式数据集，便于扩展新的评测数据
- **可扩展的prompt模板系统**：内置20+种主流模型prompt模板，支持快速适配不同模型
- **分布式结果收集器**：自动按benchmark分类保存结果，支持细粒度分析

## 📦 快速开始

### 安装

```bash
git clone https://github.com/Open-DataFlow/Math-Eval.git
cd Math-Eval
pip install -r requirements.txt
```

### 基础评估

运行基础数学能力评估：

```bash
python eval.py --config examples/example.yaml
```

### 分类评估

运行带有类别信息的详细评估：

```bash
python eval_category.py --config examples/example.yaml
```

### 配置文件说明

配置文件采用YAML格式，主要包含以下参数：

```yaml
# 评估的benchmark列表，用逗号分隔
benchmarks: "aime24,gsm8k,math,amc23,olympiadbench,gaokao2024_mix,minerva_math"

# 任务名称，用于结果文件命名
task_name: test

# 模型配置参数
model_args:
  model_path: /path/to/your/model  # 模型路径
  temperature: 0.0                 # 采样温度
  top_p: 0.95                     # 核采样参数
  top_k: -1                       # Top-K采样
  max_model_len: 16384            # 模型最大长度
  max_tokens: 16384               # 生成最大token数
  tensor_parallel_size: 1         # 张量并行大小
  pipeline_parallel_size: 8       # 流水线并行大小

# 验证方法配置
verify_method: "qwen_eval"        # 可选: qwen_eval, xverify
prompt_template: "qwen25-math-cot" # prompt模板名称

# xVerify验证器配置（当verify_method为xverify时使用）
xverify_args:
  inference_mode: "custom"
  model_name: ""
  model_path_or_url: ""
  api_key: ""
  process_num: 4
```

## 📊 支持的数据集

Math-Eval支持多种主流数学评测数据集：

| 数据集 | 描述 | 题目数量 | 难度级别 |
|--------|------|----------|----------|
| GSM8K | 小学数学应用题 | 1,319 | 初级 |
| MATH | 高中数学竞赛题 | 5,000 | 高级 |
| AIME24 | 美国数学邀请赛2024 | 30 | 竞赛级 |
| AMC23 | 美国数学竞赛2023 | 25 | 竞赛级 |
| OlympiadBench | 数学奥林匹克题目 | 8,476 | 竞赛级 |
| Gaokao2024_mix | 2024年高考数学题 | 390 | 中高级 |
| Minerva_math | Minerva数学数据集 | 1,000 | 高级 |

## 🔧 高级功能

### 自定义Prompt模板

框架内置了20+种prompt模板，支持主流大语言模型：

- `qwen25-math-cot`: Qwen2.5-Math思维链模板
- `deepseek-math`: DeepSeek-Math专用模板  
- `numina`: Numina模型格式
- `mathstral`: Mathstral模型格式
- `direct`: 直接问答格式
- `cot`: 通用思维链格式

### 验证方法详解

#### 规则验证 (qwen_eval)
使用标准化的数值验证流程，适合大多数数学问题：
- 自动提取答案中的数值
- 支持分数、小数、百分比等多种格式
- 处理单位换算和数值近似

#### 模型验证 (xverify)
使用另一个模型作为验证器，适合复杂推理问题：
- 支持本地模型和API调用
- 可配置验证模型参数
- 提供更灵活的验证策略

### 结果分析

评估完成后，系统会生成详细的结果报告：

```
results/
└── test_20241214_120000/
    ├── results.json              # 汇总结果
    ├── gsm8k_results.jsonl      # GSM8K详细结果
    ├── math_results.jsonl       # MATH详细结果
    └── ...                      # 其他benchmark结果
```

#### 基础评估结果格式
```json
[
  {
    "benchmark": "gsm8k",
    "accuracy": 0.85
  },
  {
    "benchmark": "math", 
    "accuracy": 0.42
  }
]
```

#### 分类评估结果格式
```json
{
  "General": {
    "Algebra": 0.78,
    "Geometry": 0.65,
    "Number Theory": 0.72
  },
  "gsm8k": {
    "overall": 0.85,
    "by_category": {
      "Algebra": 0.88,
      "Arithmetic": 0.92
    }
  }
}
```

## 🛠️ 开发指南

### 添加新的数据集

1. 准备JSON Lines格式的数据文件：
```json
{"question": "问题内容", "answer": "标准答案", "primary_category": "类别"}
```

2. 将文件放置在`benchmarks/`目录下

3. 在配置文件中添加数据集名称

### 自定义Prompt模板

在`src/prompts.py`中添加新的模板：

```python
PROMPT_TEMPLATES["your_template"] = (
    "输入模板: {input}",
    "输出模板: {output}", 
    "分隔符"
)
```

### 扩展验证方法

实现新的验证器类，继承基础验证接口：

```python
class CustomVerifier:
    def verify_from_input(self, problems, solutions, answers):
        # 实现验证逻辑
        return results
```

## 📈 性能优化

### GPU内存优化
- 调整`tensor_parallel_size`和`pipeline_parallel_size`参数
- 根据GPU显存调整`max_model_len`
- 使用量化模型减少内存占用

### 推理速度优化  
- 启用vLLM的KV缓存
- 批量处理多个问题
- 合理设置`max_tokens`避免过长生成

### 评估效率优化
- 使用规则验证替代模型验证（当准确性要求不高时）
- 并行处理多个benchmark
- 缓存模型加载状态

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

### 贡献类型
- 新增数据集支持
- 优化推理性能
- 改进验证算法
- 完善文档说明
- 修复bug

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [vLLM](https://github.com/vllm-project/vllm) - 高性能推理引擎
- [math_verify](https://github.com/QwenLM/Qwen2.5-Math) - 数学验证库
- [xVerify](https://github.com/GAIR-NLP/xVerify) - 智能验证框架

## 📞 联系我们

如有问题或建议，请通过以下方式联系：

- 提交Issue: [GitHub Issues](https://github.com/Open-DataFlow/Math-Eval/issues)
- 讨论区: [GitHub Discussions](https://github.com/Open-DataFlow/Math-Eval/discussions)

---

**Math-Eval** - 让大模型数学能力评估更简单、更高效！

