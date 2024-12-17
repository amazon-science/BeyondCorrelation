import argparse
import copy
import json
import logging
import os
import random
import urllib.request
from multiprocessing import Pool

import tqdm

from beyond_correlation.bedrock_utils import llm_configs
from beyond_correlation.beyond_correlation_compare import (
    beyond_correlation_for_ordinal_data,
    beyond_correlation_for_categorical_data,
)

DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/dmg-illc/JUDGE-BENCH/refs/heads/master/data/"
    "recipe_crowd_sourcing_data/meta_evaluation_recipes.json"
)


def parse_score(metric, x):
    if "[" in x:
        x = x[x.find("[") + 1 :]
    if "]" in x:
        x = x[: x.find("]")]
    x = x.strip()

    if metric["category"] == "categorical":
        if x in metric["labels_list"]:
            return x
        for t in metric["labels_list"]:
            if t in x:
                return t
        print(f"Cannot parse {x}.")
        return None
    elif metric["category"] == "graded":
        if x.isdigit() and metric["worst"] <= int(x) <= metric["best"]:
            return int(x)
        for t in range(metric["worst"], metric["best"] + 1):
            if str(t) in x:
                return t
        print(f"Cannot parse {x}.")
        return None
    else:
        raise ValueError(f"{metric['category']} not supported.")


def process_dataset(dataset):
    metadata = []
    prompts = []

    for metric in dataset["annotations"]:
        for i, example in enumerate(dataset["instances"]):
            metadata.append({"metric": metric, "example_index": i})
            prompt_text = metric["prompt"]
            if isinstance(example["instance"], dict):
                for field, value in example["instance"].items():
                    assert "{{ %s }}" % field in prompt_text
                    prompt_text = prompt_text.replace("{{ %s }}" % field, value)
            else:
                prompt_text = prompt_text.replace("{{ instance }}", example["instance"])
            assert "{{" not in prompt_text and "}}" not in prompt_text
            if metric["category"] == "categorical":
                response_options = ", ".join([f"[{x}]" for x in metric["labels_list"]])
            elif metric["category"] == "graded":
                assert metric["worst"] < metric["best"]
                response_options = ", ".join([f"[{x}]" for x in range(metric["worst"], metric["best"] + 1)])
            else:
                raise ValueError(f"{metric['category']} not supported.")
            prompt_text += (
                f"\n\nChoose your answer from {response_options}. "
                f"Your answer must be an score with squared brackets. "
                f"Never write anything else."
            )

            prompts.append(prompt_text)
    return metadata, prompts


def run_judge_bench_inference(dataset, llm_name, num_workers=4):
    llm_predict_fn = llm_configs[llm_name]
    metadata, prompts = process_dataset(dataset)

    with Pool(num_workers) as p:
        responses = list(tqdm.tqdm(p.imap(llm_predict_fn, prompts), total=len(prompts)))

    for meta, response in zip(metadata, responses):
        example = dataset["instances"][meta["example_index"]]
        if llm_name + "_raw" not in example:
            example[llm_name + "_raw"] = {}
        example[llm_name + "_raw"][meta["metric"]["metric"]] = response

    metric_info = {}
    for metric in dataset["annotations"]:
        metric_info[metric["metric"]] = metric

    for example in dataset["instances"]:
        example[llm_name] = {}
        for metric in example[llm_name + "_raw"]:
            example[llm_name][metric] = [
                parse_score(metric_info[metric], x) for x in example[llm_name + "_raw"][metric]
            ]


def filter_none(labels):
    return [x for x in labels if x is not None]


def get_labels_for_metric(metric, dataset, llm):
    human_labels = []
    machine_labels = []
    metric_name = metric["metric"]
    for example in dataset["instances"]:
        human_labels_example = copy.deepcopy(example["annotations"][metric_name]["individual_human_scores"])
        machine_labels_example = copy.deepcopy(filter_none(example[llm][metric_name]))
        random.shuffle(human_labels_example)
        random.shuffle(machine_labels_example)
        if len(human_labels_example) > 0 and len(machine_labels_example) > 0:
            human_labels.append(human_labels_example)
            machine_labels.append(machine_labels_example)
        else:
            logging.warning("missing annotation")
    return human_labels, machine_labels


def run_judge_bench_analysis(dataset, llm_name, exp_dir, dataset_name):
    for metric in dataset["annotations"]:
        human_labels, machine_labels = get_labels_for_metric(metric, dataset, llm_name)
        result_dir = os.path.join(exp_dir, f"result_{metric['metric'].replace(' ', '_')}")
        os.makedirs(result_dir, exist_ok=True)
        if metric["category"] == "categorical":
            beyond_correlation_for_categorical_data(
                human_labels=human_labels,
                machine_labels=machine_labels,
                unique_label_list=metric["labels_list"],
                result_dir=result_dir,
                experiment_name=f"{dataset_name} - {metric['metric']}",
                majority_percentage_bins=[(0, 40), (40, 60), (60, 80), (80, 100)],
            )
        elif metric["category"] == "graded":
            beyond_correlation_for_ordinal_data(
                human_labels=human_labels,
                machine_labels=machine_labels,
                ordinal_min=metric["worst"],
                ordinal_max=metric["best"],
                result_dir=result_dir,
                experiment_name=f"{dataset_name} - {metric['metric']}",
                median_percentage_bins=[(0, 40), (40, 60), (60, 80), (80, 100)],
            )
        else:
            raise ValueError(f"{metric['category']} not supported.")


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser("Example using datasets from `github.com/dmg-illc/JUDGE-BENCH`")
    parser.add_argument("--dataset_name", default="Recipes", help="Dataset name shown in the figure title.")
    parser.add_argument(
        "--dataset_path",
        help="please prepare dataset using scripts in `github.com/dmg-illc/JUDGE-BENCH`, "
        "and add the path to the JSON file. By default, we will use "
        "`github.com/dmg-illc/JUDGE-BENCH/blob/master/data/recipe_crowd_sourcing_data/meta_evaluation_recipes.json`",
    )
    parser.add_argument("--exp_dir", default="judge_bench_recipe_experiment", help="directory for the experiment.")
    parser.add_argument("--sample", default=0, type=int, help="subsample size. 0 for all data.")
    parser.add_argument("--seed", default=42, type=int, help="random seed for sampling.")
    parser.add_argument("--llm", default="llama_3_70B", choices=list(llm_configs.keys()), help="LLM for experiment.")
    args = parser.parse_args()

    exp_dir = args.exp_dir
    data_dir = os.path.join(exp_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    if not args.dataset_path:
        dataset_path = os.path.join(data_dir, DEFAULT_DATASET_URL.split("/")[-1])
        urllib.request.urlretrieve(DEFAULT_DATASET_URL, dataset_path)
    else:
        dataset_path = args.dataset_path
    dataset_path_with_llm_inference = os.path.join(
        data_dir, dataset_path.split("/")[-1][: -len(".json")] + f"-{args.llm}-sample{args.sample}-seed{args.seed}.json"
    )
    with open(dataset_path) as f:
        dataset = json.load(f)

    if args.sample:
        random.seed(args.seed)
        selected_indices = random.sample(range(1, len(dataset["instances"]) + 1), args.sample)
        dataset["instances"] = [dataset["instances"][i] for i in selected_indices]

    if os.path.exists(dataset_path_with_llm_inference):
        with open(dataset_path_with_llm_inference) as f:
            dataset = json.load(f)
    else:
        run_judge_bench_inference(dataset, args.llm)
        with open(dataset_path_with_llm_inference, "w") as f:
            json.dump(dataset, f, indent=2)

    random.seed(args.seed)
    run_judge_bench_analysis(dataset, args.llm, exp_dir, dataset_name=args.dataset_name)


if __name__ == "__main__":
    main()
