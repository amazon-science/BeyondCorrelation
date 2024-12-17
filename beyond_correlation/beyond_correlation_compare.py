import os
from typing import List, Tuple

import pandas as pd

from beyond_correlation.correlation_utils import compute_correlation_ordinal, compute_correlation_categorical
from beyond_correlation.dataframe_utils import transform_df_ordinal, transform_df_categorical
from beyond_correlation.jsd_visualization import visualize_hm_jsd

DEFAULT_PERCENTAGE_BINS = [(0, 60), (60, 80), (80, 100)]


def verify_annotations(labels, unique_label_list):
    for example_labels in labels:
        for label in example_labels:
            assert label in unique_label_list


def beyond_correlation_for_ordinal_data(
    human_labels: List[List[int]],
    machine_labels: List[List[int]],
    ordinal_min: int,
    ordinal_max: int,
    result_dir: str,
    experiment_name: str = "",
    median_percentage_bins: List[Tuple] = None,
) -> pd.DataFrame:
    os.makedirs(result_dir, exist_ok=True)
    unique_label_list = [label for label in range(ordinal_min, ordinal_max + 1)]
    verify_annotations(human_labels, unique_label_list)
    verify_annotations(machine_labels, unique_label_list)

    median_percentage_bins = median_percentage_bins or DEFAULT_PERCENTAGE_BINS

    df = pd.DataFrame({"human_labels": human_labels, "machine_labels": machine_labels})
    df = transform_df_ordinal(df, unique_label_list=unique_label_list)

    # plot label distributions
    visualize_hm_jsd(
        df,
        unique_label_list=unique_label_list,
        human_column_name="human_labels",
        human_aggregated_column_name="human_labels_median",
        machine_column_name="machine_labels",
        machine_aggregated_column_name="machine_labels_median",
        title=experiment_name,
        save_loc=os.path.join(result_dir, "dist_pie.pdf"),
    )

    # bucketing and compute correlations
    results = []

    df_sub = df
    result = {"name": "All", "proportion": 100 * len(df_sub) / len(df)}
    result.update(compute_correlation_ordinal(df_sub, unique_label_list))
    results.append(result)

    for num_unique in range(1, ordinal_max - ordinal_min + 2):
        df_sub = df.query(f"human_labels_num_unique == {num_unique}")
        result = {"name": f"human_labels_num_unique = {num_unique}", "proportion": 100 * len(df_sub) / len(df)}
        if num_unique == 1:
            result["name"] += " (perfect agreement)"
        result.update(compute_correlation_ordinal(df_sub, unique_label_list))
        results.append(result)

    for bucket_min, bucket_max in median_percentage_bins:
        df_sub = df.query(f"{bucket_min} <= human_labels_median_percent_vote < {bucket_max}")
        result = {
            "name": f"{bucket_min} <= human_labels_median_percent_vote < {bucket_max}",
            "proportion": 100 * len(df_sub) / len(df),
        }
        result.update(compute_correlation_ordinal(df_sub, unique_label_list))
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(result_dir, "beyond_correlation.csv"), index=False)

    return results_df


def beyond_correlation_for_categorical_data(
    human_labels: List[List[int]],
    machine_labels: List[List[int]],
    unique_label_list: List[str],
    result_dir: str,
    experiment_name: str = "",
    majority_percentage_bins: List[Tuple] = None,
) -> pd.DataFrame:
    os.makedirs(result_dir, exist_ok=True)
    verify_annotations(human_labels, unique_label_list)
    verify_annotations(machine_labels, unique_label_list)

    majority_percentage_bins = majority_percentage_bins or DEFAULT_PERCENTAGE_BINS

    df = pd.DataFrame({"human_labels": human_labels, "machine_labels": machine_labels})
    df = transform_df_categorical(df, unique_label_list=unique_label_list)

    # plot label distributions
    visualize_hm_jsd(
        df,
        unique_label_list=unique_label_list,
        human_column_name="human_labels",
        human_aggregated_column_name="human_labels_majority",
        machine_column_name="machine_labels",
        machine_aggregated_column_name="machine_labels_majority",
        bin_type="majority",
        title=experiment_name,
        save_loc=os.path.join(result_dir, "dist_pie.pdf"),
    )

    # bucketing and compute correlations
    results = []

    df_sub = df
    result = {"name": "All", "proportion": 100 * len(df_sub) / len(df)}
    result.update(compute_correlation_categorical(df_sub, unique_label_list))
    results.append(result)

    for n_unique in range(1, len(unique_label_list) + 1):
        df_sub = df.query(f"human_labels_num_unique == {n_unique}").copy()
        result = {"name": f"Number of unique human labels (u={n_unique})", "proportion": 100 * len(df_sub) / len(df)}
        result.update(compute_correlation_categorical(df_sub, unique_label_list))
        if n_unique == 1:
            result["name"] += " (perfect agreement)"
        results.append(result)

    for bucket_min, bucket_max in majority_percentage_bins:
        df_sub = df.query(f"{bucket_min} <= human_labels_majority_percentage_vote < {bucket_max}").copy()
        result = {
            "name": f"{bucket_min}% <= Human label majority vote < {bucket_max}%",
            "proportion": 100 * len(df_sub) / len(df),
        }
        result.update(compute_correlation_categorical(df_sub, unique_label_list))
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(result_dir, "beyond_correlation.csv"), index=False)
    return results_df
