# evaluation.py
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

def main():
    """
    Step 1. 读取 yaml 配置
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
    """
    benchmark_list = config["benchmarks"].split(",")
    df = pd.DataFrame()
    for benchmark in benchmark_list:
        bmk = pd.read_json(f"benchmarks/{benchmark}.jsonl",
                            orient="records", lines=True)
        bmk["benchmark"] = benchmark
        df = pd.concat([df, bmk], ignore_index=True)
    logger.info(f"Load {len(df)} examples from {benchmark_list}")

    """
    Step 3. 构造 prompt
    """
    prompt_template = config["prompt_template"]
    prompt_func = get_prompt(prompt_template)
    raw_problems = df["question"].tolist()
    inputs = [prompt_func(problem) for problem in raw_problems]

    """
    Step 4. 调用模型生成答案
    """
    model_args = config["model_args"]
    engine = VllmAsyncEngine(model_args)
    results = engine.run(inputs)
    df["model_solution"] = results

    """
    Step 5. 评估答案
    """
    gt = df["answer"].tolist()
    method = config["verify_method"]
    if method == "qwen_eval":
        logger.info("Start to verify answers with qwen_eval")
        from src.AnswerExtraction import UnitTextManager, StringCleaner, AnswerExtractor
        from math_verify import parse, verify
        unit_manager = UnitTextManager()
        string_cleaner = StringCleaner(unit_manager)
        answer_extractor = AnswerExtractor(string_cleaner)
        model_solutions = df["model_solution"].tolist()
        model_answers = [answer_extractor.extract_answer(sol, None)
                         for sol in model_solutions]
        df["model_answer"] = model_answers
        results_bool = []
        for i in range(len(model_answers)):
            golden = gt[i]
            pred = model_answers[i]
            ok = float(verify(parse(str(golden)), parse(str(pred)))) > 0
            results_bool.append(bool(ok))
        df["verify_result"] = results_bool

    elif method == "xverify":
        from src.xverifier import AnswerJudger_xverify
        xverifier = AnswerJudger_xverify(config["xverify_args"])
        res = xverifier.verify_from_input(
            df["problem"].tolist(),
            df["model_solution"].tolist(),
            gt
        )
        df["verify_result"] = res

    else:
        raise ValueError(f"Invalid verify method: {method}")

    """
    Step 6. 计算并报告 General 和每个 benchmark 的整体及分类别正确率
    """
    # 创建结果目录
    results_dir = f"results/{config['task_name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)

    # 先保存每个 benchmark 的详细结果
    for benchmark in benchmark_list:
        bmk_df = df[df["benchmark"] == benchmark]
        bmk_df.to_json(
            os.path.join(results_dir, f"{benchmark}_results.jsonl"),
            orient="records",
            lines=True,
            force_ascii=False
        )

    # 获取所有可能的类别
    categories = sorted(df["primary_category"].dropna().unique())

    # 1) General: 全部数据中，每个类别的正确率
    general_acc = {}
    for cat in categories:
        cat_df = df[df["primary_category"] == cat]
        if len(cat_df) > 0:
            general_acc[cat] = cat_df["verify_result"].mean()
        else:
            general_acc[cat] = None  # 不太可能，但为了保险

    # 2) 每个 benchmark：整体 & 分类别正确率
    benchmark_report = {}
    for benchmark in benchmark_list:
        bmk_df = df[df["benchmark"] == benchmark]
        # 整体正确率
        overall = bmk_df["verify_result"].mean() if len(bmk_df) > 0 else None

        # 分类别正确率
        by_cat = {}
        for cat in categories:
            subset = bmk_df[bmk_df["primary_category"] == cat]
            if len(subset) > 0:
                by_cat[cat] = subset["verify_result"].mean()
            else:
                by_cat[cat] = None

        benchmark_report[benchmark] = {
            "overall": overall,
            "by_category": by_cat
        }

    # 3) 合并成最终的 JSON 报告
    report = {
        "General": general_acc,
        **benchmark_report
    }

    # 4) 写入 disk
    with open(
        os.path.join(results_dir, "results.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    logger.info(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()
