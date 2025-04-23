# Math-Eval - 大模型数学能力高效评估框架

[English](./README_EN.md) | 中文简体

## 🚀 核心优势
1. **高效并行推理引擎**
   - 基于vLLM异步调用引擎，支持张量并行（Tensor Parallelism）与数据并行（Data Parallelism）的混合加速策略
   - 多benchmark评估时只需单次启动vLLM引擎，最大限度提升GPU利用率

2. **双重答案验证机制**
   - **规则验证**：集成Qwen-Math的答案提取方法与math_verify库的标准化校验流程
   - **模型验证**：基于xVerify框架支持本地模型或API服务的智能验证
   - 支持自定义验证策略组合（AND/OR逻辑组合）

3. **模块化设计**
   - 可插拔式benchmark加载器（JSONL格式）
   - 可扩展的prompt模板系统
   - 分布式结果收集器

## 📦 安装
```bash
git clone https://github.com/Open-DataFlow/Math-Eval.git
cd Math-Eval
pip install -r requirements.txt
```