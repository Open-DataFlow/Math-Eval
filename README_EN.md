# Math-Eval - Efficient Mathematical Capability Evaluation Framework for Large Language Models

[中文简体](README.md) | English

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![vLLM](https://img.shields.io/badge/vLLM-0.4.1+-green.svg)](https://github.com/vllm-project/vllm)

Math-Eval is a high-performance evaluation framework specifically designed for assessing the mathematical capabilities of large language models. The framework integrates advanced parallel inference engines, dual answer verification mechanisms, and modular design to provide researchers and developers with a comprehensive and reliable mathematical capability assessment tool.

## 🚀 Key Features

### 1. High-Performance Parallel Inference Engine
Built on vLLM's asynchronous inference engine, supporting hybrid acceleration strategies with Tensor Parallelism and Data Parallelism. Multiple benchmark evaluations require only a single vLLM engine startup, maximizing GPU utilization and significantly reducing evaluation time costs.

### 2. Dual Answer Verification Mechanism
- **Rule-based Verification**: Integrates Qwen-Math's answer extraction methods with math_verify library's standardized numerical verification process, ensuring accuracy of numerical computations
- **Model-based Verification**: Based on xVerify framework supporting local models or API services for intelligent verification, providing more flexible verification strategies
- **Customizable Verification Strategy Combinations**: Configurable AND/OR logic combinations to meet different evaluation requirements

### 3. Modular Design
- **Pluggable Benchmark Loader**: Supports JSON format datasets for easy extension of new evaluation data
- **Extensible Prompt Template System**: Built-in 20+ mainstream model prompt templates for quick adaptation to different models
- **Distributed Result Collector**: Automatically saves results by benchmark classification, supporting fine-grained analysis

## 📦 Quick Start

### Installation

```bash
git clone https://github.com/Open-DataFlow/Math-Eval.git
cd Math-Eval
pip install -r requirements.txt
```

### Basic Evaluation

Run basic mathematical capability evaluation:

```bash
python eval.py --config examples/example.yaml
```

### Category-based Evaluation

Run detailed evaluation with category information:

```bash
python eval_category.py --config examples/example.yaml
```

### Configuration File Description

Configuration files use YAML format and mainly include the following parameters:

```yaml
# List of benchmarks to evaluate, separated by commas
benchmarks: "aime24,gsm8k,math,amc23,olympiadbench,gaokao2024_mix,minerva_math"

# Task name for result file naming
task_name: test

# Model configuration parameters
model_args:
  model_path: /path/to/your/model  # Model path
  temperature: 0.0                 # Sampling temperature
  top_p: 0.95                     # Nucleus sampling parameter
  top_k: -1                       # Top-K sampling
  max_model_len: 16384            # Maximum model length
  max_tokens: 16384               # Maximum generation tokens
  tensor_parallel_size: 1         # Tensor parallel size
  pipeline_parallel_size: 8       # Pipeline parallel size

# Verification method configuration
verify_method: "qwen_eval"        # Options: qwen_eval, xverify
prompt_template: "qwen25-math-cot" # Prompt template name

# xVerify verifier configuration (used when verify_method is xverify)
xverify_args:
  inference_mode: "custom"
  model_name: ""
  model_path_or_url: ""
  api_key: ""
  process_num: 4
```

## 📊 Supported Datasets

Math-Eval supports various mainstream mathematical evaluation datasets:

| Dataset | Description | Number of Problems | Difficulty Level |
|---------|-------------|-------------------|------------------|
| GSM8K | Elementary math word problems | 1,319 | Elementary |
| MATH | High school math competition problems | 5,000 | Advanced |
| AIME24 | American Invitational Mathematics Examination 2024 | 30 | Competition |
| AMC23 | American Mathematics Competitions 2023 | 25 | Competition |
| OlympiadBench | Mathematical Olympiad problems | 8,476 | Competition |
| Gaokao2024_mix | 2024 Chinese College Entrance Exam math problems | 390 | Intermediate-Advanced |
| Minerva_math | Minerva mathematics dataset | 1,000 | Advanced |

## 🔧 Advanced Features

### Custom Prompt Templates

The framework includes 20+ built-in prompt templates supporting mainstream large language models:

- `qwen25-math-cot`: Qwen2.5-Math chain-of-thought template
- `deepseek-math`: DeepSeek-Math specialized template  
- `numina`: Numina model format
- `mathstral`: Mathstral model format
- `direct`: Direct question-answer format
- `cot`: General chain-of-thought format

### Verification Methods Explained

#### Rule-based Verification (qwen_eval)
Uses standardized numerical verification process, suitable for most mathematical problems:
- Automatically extracts numerical values from answers
- Supports multiple formats including fractions, decimals, percentages
- Handles unit conversions and numerical approximations

#### Model-based Verification (xverify)
Uses another model as a verifier, suitable for complex reasoning problems:
- Supports local models and API calls
- Configurable verification model parameters
- Provides more flexible verification strategies

### Result Analysis

After evaluation completion, the system generates detailed result reports:

```
results/
└── test_20241214_120000/
    ├── results.json              # Summary results
    ├── gsm8k_results.jsonl      # GSM8K detailed results
    ├── math_results.jsonl       # MATH detailed results
    └── ...                      # Other benchmark results
```

#### Basic Evaluation Result Format
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

#### Category-based Evaluation Result Format
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

## 🛠️ Development Guide

### Adding New Datasets

1. Prepare data file in JSON Lines format:
```json
{"question": "Problem content", "answer": "Standard answer", "primary_category": "Category"}
```

2. Place the file in the `benchmarks/` directory

3. Add the dataset name to the configuration file

### Custom Prompt Templates

Add new templates in `src/prompts.py`:

```python
PROMPT_TEMPLATES["your_template"] = (
    "Input template: {input}",
    "Output template: {output}", 
    "Separator"
)
```

### Extending Verification Methods

Implement new verifier classes inheriting from the base verification interface:

```python
class CustomVerifier:
    def verify_from_input(self, problems, solutions, answers):
        # Implement verification logic
        return results
```

## 📈 Performance Optimization

### GPU Memory Optimization
- Adjust `tensor_parallel_size` and `pipeline_parallel_size` parameters
- Adjust `max_model_len` based on GPU memory
- Use quantized models to reduce memory usage

### Inference Speed Optimization  
- Enable vLLM's KV cache
- Batch process multiple problems
- Set reasonable `max_tokens` to avoid overly long generation

### Evaluation Efficiency Optimization
- Use rule-based verification instead of model verification (when accuracy requirements are not high)
- Parallel processing of multiple benchmarks
- Cache model loading state

## 🤝 Contributing

We welcome community contributions! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Contribution Types
- Adding new dataset support
- Optimizing inference performance
- Improving verification algorithms
- Enhancing documentation
- Bug fixes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [vLLM](https://github.com/vllm-project/vllm) - High-performance inference engine
- [math_verify](https://github.com/QwenLM/Qwen2.5-Math) - Mathematical verification library
- [xVerify](https://github.com/GAIR-NLP/xVerify) - Intelligent verification framework

## 📞 Contact Us

For questions or suggestions, please contact us through:

- Submit Issues: [GitHub Issues](https://github.com/Open-DataFlow/Math-Eval/issues)
- Discussions: [GitHub Discussions](https://github.com/Open-DataFlow/Math-Eval/discussions)

---

**Math-Eval** - Making large language model mathematical capability evaluation simpler and more efficient!

