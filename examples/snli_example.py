import argparse
import json
import logging
import os
import random
from functools import partial
from multiprocessing import Pool

import tqdm
from jsonlines import jsonlines

from beyond_correlation.bedrock_utils import llm_configs
from beyond_correlation.beyond_correlation_compare import beyond_correlation_for_categorical_data

PROMPT = """You will be presented with a premise and a hypothesis about that premise. \
You need to decide whether the hypothesis is entailed by the premise by choosing one of the following answers:

[[e]]: The hypothesis follows logically from the information contained in the premise.
[[n]]: It is not possible to determine whether the hypothesis is true or false without further information.
[[c]]: The hypothesis is logically false from the information contained in the premise.

Read the following premise and hypothesis thoroughly and select the correct answer from the three answer labels.

Premise:
{premise}

Hypothesis:
{hypothesis}

Make a selection from "[[e]]", "[[n]]", "[[c]]". Only write the answer, do not write reasons. """


def nli_llm_predict_fn(example, llm_predict_fn):
    if "premise" in example:
        premise, hypothesis = example["premise"], example["hypothesis"]
    elif "sentence1" in example:
        premise, hypothesis = example["sentence1"], example["sentence2"]
    else:
        raise ValueError
    return llm_predict_fn(PROMPT.format(premise=premise, hypothesis=hypothesis))


def filter_none(labels):
    return [x for x in labels if x is not None]


def parse_nli_response(value):
    mappers = {"n": "neutral", "e": "entailment", "c": "contradiction"}
    value_clean = value.strip().lstrip('["').rstrip(']"').lower()
    if value_clean in mappers:
        return mappers[value_clean]
    else:
        logging.warning(f"Cannot parse {json.dumps(value)}")
    return None


def run_snli_inference(dataset, llm_name, num_workers=4):
    llm_predict_fn = partial(nli_llm_predict_fn, llm_predict_fn=llm_configs[llm_name])

    with Pool(num_workers) as p:
        all_response = list(tqdm.tqdm(p.imap(llm_predict_fn, dataset), total=len(dataset)))
    for i in range(len(dataset)):
        dataset[i][llm_name + "_raw"] = all_response[i]
        dataset[i][llm_name] = filter_none([parse_nli_response(x) for x in all_response[i]])


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="please download data from `nlp.stanford.edu/projects/snli/snli_1.0.zip` and "
             "provide the path to one of the jsonl file.",
    )
    parser.add_argument("--exp_dir", default="snli_experiment", help="directory for the experiment.")
    parser.add_argument("--sample", default=100, type=int, help="subsample size. 0 for all data.")
    parser.add_argument("--seed", default=42, type=int, help="random seed for sampling.")
    parser.add_argument("--llm", default="llama_3_70B", choices=list(llm_configs.keys()), help="LLM for experiment.")
    args = parser.parse_args()

    exp_dir = args.exp_dir
    data_dir = os.path.join(exp_dir, "data")
    result_dir = os.path.join(exp_dir, "result")
    os.makedirs(data_dir, exist_ok=True)

    with jsonlines.open(args.dataset_path, "r") as f:
        dataset = list(f)
    dataset_path_with_llm_inference = os.path.join(
        data_dir,
        args.dataset_path.split("/")[-1][: -len(".jsonl")] + f"-{args.llm}-sample{args.sample}-seed{args.seed}.json",
    )

    if os.path.exists(dataset_path_with_llm_inference):
        with jsonlines.open(dataset_path_with_llm_inference, "r") as f:
            dataset = list(f)
    else:
        if args.sample:
            random.seed(args.seed)
            selected_indices = random.sample(range(1, len(dataset) + 1), args.sample)
            dataset = [dataset[i] for i in selected_indices]
        run_snli_inference(dataset, args.llm)
        with jsonlines.open(dataset_path_with_llm_inference, "w") as f:
            f.write_all(dataset)

    human_labels = []
    machine_labels = []
    for item in dataset:
        if len(item["annotator_labels"]) and len(item[args.llm]):
            human_labels.append(item["annotator_labels"])
            machine_labels.append(item[args.llm])
        else:
            logging.warning("missing annotataion.")

    beyond_correlation_for_categorical_data(
        human_labels=human_labels,
        machine_labels=machine_labels,
        unique_label_list=["entailment", "contradiction", "neutral"],
        result_dir=result_dir,
        experiment_name="SNLI",
        majority_percentage_bins=[(0, 60), (60, 80), (80, 100)],
    )


if __name__ == "__main__":
    main()
