# 主函数，从yaml中读取所有的参数，并执行完整的evaluation逻辑
import yaml
import os
import sys
import argparse
import logging
import pandas as pd
import time
import json
from src.prompts import get_prompt
from src.VllmAsyncEngine import VllmAsyncEngine
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
Step 1. 读取yaml
"""
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="examples/example.yaml")
args = parser.parse_args()

yaml_path = args.config
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

logger.info(f"Load config from {yaml_path}")
logger.info(f"Config: {config}")


"""
Step 2. 读取数据
从config读取所有需要eval的benchmark，将数据加载到dataframe中
"""
benchmark_list = config["benchmarks"].split(",")
df = pd.DataFrame()
for benchmark in benchmark_list:
    bmk = pd.read_json(f"bmk_category/modified_{benchmark}.jsonl", orient="records", lines=True)
    bmk["benchmark"] = benchmark
    df = pd.concat([df, bmk])

logger.info(f"Load {len(df)} examples from {benchmark_list}")


"""
Step 3. 读取prompt template
从config读取prompt template，并根据prompt template生成prompt
"""

prompt_template = config["prompt_template"]
prompt_func = get_prompt(prompt_template)
raw_problems = df["question"].tolist()
inputs = [prompt_func(problem) for problem in raw_problems]


"""
Step 4. 生成答案
生成答案，并写入dataframe
"""
model_args = config["model_args"]
engine = VllmAsyncEngine(model_args)
results = engine.run(inputs)
df["model_solution"] = results

"""
Step 5. 评估答案
根据config中的verify_method，评估答案
"""
gt = df["answer"].tolist()
method = config["verify_method"]
if method == "qwen_eval":
    logger.info("Start to verify answers with qwen_eval")
    from src.AnswerExtraction import UnitTextManager, StringCleaner, AnswerExtractor
    from math_verify import parse,verify
    unit_manager = UnitTextManager()
    string_cleaner = StringCleaner(unit_manager)
    answer_extractor = AnswerExtractor(string_cleaner)
    model_solutions = df["model_solution"].tolist()
    model_answers = [answer_extractor.extract_answer(solution,None) for solution in model_solutions]
    df["model_answer"] = model_answers
    results = []
    for i in range(len(model_answers)):
        golden_answer = gt[i]
        model_answer = model_answers[i]
        result = bool(float(verify(parse(str(golden_answer)), parse(str(model_answer)))) > 0)
        results.append(result)
    df["verify_result"] = results

elif method == "xverify":
    from src.xverifier import AnswerJudger_xverify
    xverifier = AnswerJudger_xverify(config["xverify_args"])
    results = xverifier.verify_from_input(df["problem"].tolist(), df["model_solution"].tolist(), gt)
    df["verify_result"] = results

else:
    raise ValueError(f"Invalid verify method: {method}, method should be one of [qwen_eval, xverify]")


"""
Step 6. 分割Benchmark，并计算accuracy
"""
# 在results下新建文件夹，文件夹名为task_name + 当前时间
results_dir = f"results/{config['task_name']}_{time.strftime('%Y%m%d_%H%M%S')}"
result_json = []
os.makedirs(results_dir, exist_ok=True)

# 将df按照benchmark分割
for benchmark in benchmark_list:
    bmk_df = df[df["benchmark"] == benchmark]
    bmk_df.to_json(os.path.join(results_dir, f"{benchmark}_results.jsonl"), orient="records", lines=True, force_ascii=False)
    accuracy = bmk_df["verify_result"].mean()
    rdf = {"benchmark": benchmark, "accuracy": accuracy}
    result_json.append(rdf)

# 将result_json保存为json文件
with open(os.path.join(results_dir, "results.json"), "w") as f:
    json.dump(result_json, f, ensure_ascii=False, indent=4)

logger.info(f"Results saved to {results_dir}")