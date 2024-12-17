import argparse
import json
import logging
import os.path
import urllib.request

import pandas as pd

from beyond_correlation.beyond_correlation_compare import beyond_correlation_for_ordinal_data


def download_summ_eval(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    # Data source:
    # github.com/Yale-LILY
    # github.com/nlpyang/geval
    dataset_files = [
        "https://storage.googleapis.com/sfr-summarization-repo-research/model_annotations.aligned.jsonl",
        "https://raw.githubusercontent.com/nlpyang/geval/main/results/gpt4_flu_detailed.json",
        "https://raw.githubusercontent.com/nlpyang/geval/main/results/gpt4_con_detailed.json",
        "https://raw.githubusercontent.com/nlpyang/geval/main/results/gpt4_rel_detailed.json",
        "https://raw.githubusercontent.com/nlpyang/geval/main/results/gpt4_coh_detailed.json",
    ]
    for url in dataset_files:
        filename = os.path.join(data_dir, url.split("/")[-1])
        if os.path.exists(filename):
            logging.info(f"{filename} exists.")
        else:
            logging.info(f"Downloading {url}")
            urllib.request.urlretrieve(url, filename)


def parse_geval_response(all_responses):
    parsed_results = []
    for response_raw in all_responses:
        try:
            response = response_raw.strip()[:1]
            parsed_results.append(int(float(response)))
        except Exception as e:
            logging.info(f"Count not parse {json.dumps(response_raw)}.")
            continue
    return parsed_results


def map_fluency_from_5scale_to_3scale(x):
    if x < 3:
        return 1
    elif x > 3:
        return 3
    return 2


def load_summ_eval_data(data_dir):
    df_human_ann = pd.read_json(os.path.join(data_dir, "model_annotations.aligned.jsonl"), lines=True).set_index(
        ["id", "model_id"]
    )
    geval_filenames = {
        "fluency": "gpt4_flu_detailed.json",
        "consistency": "gpt4_con_detailed.json",
        "relevance": "gpt4_rel_detailed.json",
        "coherence": "gpt4_coh_detailed.json",
    }

    all_tables = {}
    for criteria, filename in geval_filenames.items():
        with open(os.path.join(data_dir, filename)) as f:
            data = json.load(f)

        df_geval = []
        for r in data:
            df_geval.append(
                {
                    "id": r["doc_id"],
                    "model_id": r["system_id"],
                    "source": r["source"],
                    "reference": r["reference"],
                    "system_output": r["system_output"],
                    "machine_labels": parse_geval_response(r["all_responses"]),
                    "criteria": criteria,
                }
            )
        df = pd.DataFrame(df_geval)

        def get_human_labels(row):
            human_labels = [ann[criteria] for ann in df_human_ann.loc[(row.id, row.model_id)]["expert_annotations"]]
            if criteria == "fluency":
                human_labels = [map_fluency_from_5scale_to_3scale(x) for x in human_labels]
            return human_labels

        df["human_labels"] = df.apply(get_human_labels, axis=1)

        all_tables[criteria] = df
    return all_tables


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="summeval_experiment", help="directory for the experiment.")
    args = parser.parse_args()
    exp_dir = args.exp_dir
    download_summ_eval(os.path.join(exp_dir, "data"))
    all_tables = load_summ_eval_data(os.path.join(exp_dir, "data"))

    for criteria, df in all_tables.items():
        df = df[(~df.human_labels.isnull()) & (~df.machine_labels.isnull())]
        beyond_correlation_for_ordinal_data(
            df.human_labels.tolist(),
            df.machine_labels.tolist(),
            ordinal_min=1,
            ordinal_max=3 if criteria == "fluency" else 5,
            result_dir=os.path.join(exp_dir, f"result_{criteria}"),
            experiment_name=f"SummEval - {criteria} (Human v.s. GEval)",
            median_percentage_bins=[(0, 60), (60, 100)],
        )


if __name__ == "__main__":
    main()
