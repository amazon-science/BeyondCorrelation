import statistics
from collections import Counter

import numpy as np


def flatten_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def add_column_num_unique(df, col):
    """Count the number of unique for a column."""
    df[f"{col}_num_unique"] = df[col].apply(lambda x: len(set([r for r in x])))
    return df


def add_column_machine_random(df, column_name, unique_label_list, num_random_label=5):
    """Create random labels."""
    machine_random_column = []
    for i in range(len(df)):
        machine_random_column.append([np.random.choice(unique_label_list) for _ in range(num_random_label)])
    df[column_name] = machine_random_column
    return df


def add_column_median(df, col):
    """Get median for labels."""
    df[f"{col}_median"] = df[col].apply(lambda x: int(statistics.median(x)))
    df[f"{col}_median_percent_vote"] = df[col].apply(
        lambda x: 100 * len([1 for xi in x if xi == int(statistics.median(x))]) / len(x)
    )
    return df


def add_column_dispersion(df, col):
    """Get dispersion (max - min) for labels."""
    df[f"{col}_dispersion"] = df[col].apply(lambda x: max(x) - min(x))
    return df


def add_column_mean(df, col):
    """Get mean for labels."""
    df[f"{col}_mean"] = df[col].apply(lambda x: statistics.mean(x))
    return df


def add_column_majority_value(df, col):
    df[f"{col}_majority"] = df[col].apply(lambda x: Counter([r for r in x]).most_common()[0][0])

    df[f"{col}_majority_percentage_vote"] = df[col].apply(
        lambda x: Counter([r for r in x]).most_common()[0][1] / len(x) * 100
    )
    return df


def _percentage_agreement_single_item(ratings_list):
    """
    This percentage calculation is obtained from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/
    Inter-rater reliability: the kappa statistic
    """
    if len(ratings_list) <= 1:
        return 0

    # if just 2 items, they must be the same else 0
    if len(ratings_list) == 2:
        if len(set(ratings_list)) == 1:
            return 1
        else:
            return 0

    # More than 2 items, return frequency of most common items/ len
    return Counter(ratings_list).most_common()[0][1] / len(ratings_list)


def compute_percentage_agreement(df, col):
    if len(df) == 0:
        return None

    return df[col].apply(_percentage_agreement_single_item).mean()


def transform_df_ordinal(df, unique_label_list, num_random_label=5):
    return (
        df.pipe(add_column_num_unique, "human_labels")
        .pipe(add_column_num_unique, "machine_labels")
        .pipe(add_column_machine_random, "random_labels", unique_label_list, num_random_label)
        .pipe(add_column_median, "random_labels")
        .pipe(add_column_median, "human_labels")
        .pipe(add_column_median, "machine_labels")
        .pipe(add_column_dispersion, "human_labels")
        .pipe(add_column_dispersion, "machine_labels")
        .pipe(add_column_mean, "human_labels")
        .pipe(add_column_mean, "machine_labels")
    )


def transform_df_categorical(df, unique_label_list, num_random_label=5):
    return (
        df.pipe(add_column_majority_value, "machine_labels")
        .pipe(add_column_num_unique, "machine_labels")
        .pipe(add_column_majority_value, "human_labels")
        .pipe(add_column_num_unique, "human_labels")
        .pipe(add_column_machine_random, "machine_random", unique_label_list, num_random_label)
        .pipe(add_column_majority_value, "machine_random")
    )
