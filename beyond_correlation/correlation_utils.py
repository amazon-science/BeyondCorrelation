import itertools
import math
import statistics
from collections import Counter

import krippendorff
import numpy as np
import scipy
from scipy.spatial import distance
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

from beyond_correlation.dataframe_utils import compute_percentage_agreement


def compute_proportion(values, labels):
    c = Counter(values)
    return [c[i] / len(values) * 100.0 for i in labels]


def compute_spearman_rank_correlation_2_column(df, column_a, column_b):
    """
    Compute krippendorff between 2 columns
    """
    if len(df) == 0:
        return None, None
    result = scipy.stats.spearmanr(df[column_a], df[column_b])
    return result.statistic, result.pvalue


def compute_cohen_2_column(df, column_a, column_b):
    return cohen_kappa_score(df[column_a], df[column_b])


def compute_krippendorff_2_column(df, column_a, column_b, level="nominal"):
    """
    Compute krippendorff between 2 columns
    """
    if len(df) == 0:
        return None

    value_counts = []
    for a, b in zip(df[column_a], df[column_b]):
        items_wise_counts = [a, b]
        value_counts.append(items_wise_counts)

    # If all values are exactly the same then cannot compute Krippendorff
    if len(np.unique(value_counts)) == 1:
        return None

    return krippendorff.alpha(reliability_data=np.array(value_counts).T, level_of_measurement=level)


def compute_krippendorff(df, col, level="nominal"):
    """
    Compute krippendorff by counting how many votes each category gets
    """
    if len(df) == 0:
        return None

    value_counts = []
    values_list = sorted(set(list(itertools.chain(*df[col].to_list()))))
    if len(values_list) == 1:
        return 1
    for h in df[col]:
        items_wise_counts = [Counter(h)[l] for l in values_list]
        value_counts.append(items_wise_counts)
    return krippendorff.alpha(value_counts=value_counts, level_of_measurement=level)


def compute_percentage_agreement_2_column(df, columna, columnb):
    """
    Compute percentage agreement
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/
    Interrater reliability: the kappa statistic ( for 2 items)
    """

    if len(df) == 0:
        return None

    # Percentage of items with exact match
    return len(df[df[columna] == df[columnb]]) / len(df)


def compute_fleiss_kappa(df, col, method="fleiss"):
    if len(df) == 0:
        return None
    label2index = {}
    max_raters = 0
    for _, item in df.iterrows():
        for label in item[col]:
            if label not in label2index:
                label2index[label] = len(label2index)
        max_raters = max(max_raters, len(item[col]))
    tab = []
    for _, item in df.iterrows():
        cnt = np.zeros(len(label2index))
        ratings = item[col]
        if len(ratings) < max_raters:
            try:
                ratings = ratings + list(np.random.choice(ratings, max_raters - len(ratings), replace=False))
            except ValueError:
                ratings = ratings + list(np.random.choice(ratings, max_raters - len(ratings), replace=True))
        for label in ratings:
            cnt[label2index[label]] += 1
        tab.append(cnt)
    tab = np.array(tab)
    return fleiss_kappa(tab, method=method)


def compute_fleiss_kappa_2_column(df, columna, columnb, method="fleiss"):
    if len(df) == 0:
        return None
    label2index = {}
    for _, item in df.iterrows():
        if item[columna] not in label2index:
            label2index[item[columna]] = len(label2index)
        if item[columnb] not in label2index:
            label2index[item[columnb]] = len(label2index)

    tab = []
    for _, item in df.iterrows():
        cnt = np.zeros(len(label2index))
        cnt[label2index[item[columna]]] += 1
        cnt[label2index[item[columnb]]] += 1
        tab.append(cnt)
    tab = np.array(tab)
    return fleiss_kappa(tab, method=method)


def compute_weighted_js(df, human_labels_column, machine_labels_column, unique_label_list, bin_type="median"):
    """
    Compute JS
        @param df: DataFrame containing the columns  human_labels_bin_col, human_labels_column, machine_labels_column
        @param human_labels_column: This is the name of the column containing human labels. Each value in this column is a list
        @param machine_labels_column: This is the name of the column containing machine labels. Each value in this column is a list
        @param bin_type: The type of binning, either median or majority depending on the type of data
    """
    result_weighted_js = 0
    if len(df) == 0:
        result_weighted_js = None
    result_bin_details = {}

    # Bin function can be median or mode ( most frequent value)
    bin_func = statistics.median if bin_type == "median" else statistics.mode

    for label in unique_label_list:
        # Assign bin number
        df_bin_number = df[human_labels_column].apply(lambda x: bin_func(x))

        # Get bin items
        df_bin_items = df[df_bin_number == label]

        if len(df_bin_items) == 0:
            bin_weight = 0
        else:
            bin_weight = len(df_bin_items) / len(df)

        result_bin_details[label] = {
            "bin_items": df_bin_items,
            "bin_number": label,
            "jsd": None,
            "bin_weight": bin_weight,
            "dist_human": [],
            "dist_machine": [],
        }

        # Skip if no items in bin
        if len(df_bin_items) == 0:
            continue

        # Compute js
        human_labels_frequencies_in_bin = Counter(list(itertools.chain(*df_bin_items[human_labels_column])))
        machine_labels_frequencies_in_bin = Counter(list(itertools.chain(*df_bin_items[machine_labels_column])))

        dist_human = [
            human_labels_frequencies_in_bin[i] / sum(Counter(human_labels_frequencies_in_bin).values())
            for i in unique_label_list
        ]

        dist_machine = [
            machine_labels_frequencies_in_bin[i] / sum(Counter(machine_labels_frequencies_in_bin).values())
            for i in unique_label_list
        ]

        result_bin_details[label]["dist_machine"] = dist_machine
        result_bin_details[label]["dist_human"] = dist_human

        jsd = distance.jensenshannon(dist_human, dist_machine)
        result_bin_details[label]["jsd"] = jsd
        result_weighted_js += bin_weight * jsd

    return result_weighted_js, result_bin_details


def _subtract_none_check(result, field_a, field_b):
    if math.isnan(result[field_a] or np.nan) or math.isnan(result[field_b] or np.nan):
        return None
    else:
        return result[field_a] - result[field_b]


def compute_correlation_ordinal(df, unique_label_list):
    result = {
        "size": len(df),
        "H_median_dispersion": df.human_labels_median.max() - df.human_labels_median.min(),
        "M_median_dispersion": df.machine_labels_median.max() - df.machine_labels_median.min(),
        "krippendorff_HH": compute_krippendorff(df, "human_labels", level="ordinal"),
        "krippendorff_MM": compute_krippendorff(df, "machine_labels", level="ordinal"),
        "krippendorff_H-median_M-median": compute_krippendorff_2_column(
            df, "human_labels_median", "machine_labels_median", level="ordinal"
        ),
        "krippendorff_HH_HM_diff": None,
        "spearman_H-median_M-median": compute_spearman_rank_correlation_2_column(
            df, "human_labels_median", "machine_labels_median"
        )[0],
        "spearman_H-mean_M-mean": compute_spearman_rank_correlation_2_column(
            df, "human_labels_mean", "machine_labels_mean"
        )[0],
        "kendall_H-median_M-median": scipy.stats.kendalltau(
            df["human_labels_median"], df["machine_labels_median"]
        ).statistic,
        "kendall_H-mean_M-mean": scipy.stats.kendalltau(df["human_labels_mean"], df["machine_labels_mean"]).statistic,
    }
    result["krippendorff_HH_HM_diff"] = _subtract_none_check(
        result, "krippendorff_HH", "krippendorff_H-median_M-median"
    )

    weighted_js, result_bin_details = compute_weighted_js(
        df, "human_labels", "machine_labels", unique_label_list=unique_label_list, bin_type="median"
    )
    result[f"weighted_JSD_median"] = weighted_js
    for label in unique_label_list:
        result[f"JSD_H_median@{label}"] = result_bin_details[label]["jsd"]

    return result


def compute_correlation_categorical_num_labeler_agnostic(df, unique_label_list):
    result = {
        "size": len(df),
        "krippendorff_HH": compute_krippendorff(df, "human_labels"),
        "krippendorff_MM": compute_krippendorff(df, "machine_labels"),
        "krippendorff_H-majority_M-majority": compute_krippendorff_2_column(
            df, "human_labels_majority", "machine_labels_majority"
        ),
        "krippendorff_H-majority_M-random": compute_krippendorff_2_column(
            df, "human_labels_majority", "machine_random_majority"
        ),
        "krippendorff_HH_HM_diff": None,
        "percentage_agreement_HH": compute_percentage_agreement(df, "human_labels"),
        "percentage_agreement_MM": compute_percentage_agreement(df, "machine_labels"),
        "percentage_agreement_H-majority_M-majority": compute_percentage_agreement_2_column(
            df, "human_labels_majority", "machine_labels_majority"
        ),
        "percentage_agreement_HH_HM_diff": None,
    }

    result["krippendorff_HH_HM_diff"] = _subtract_none_check(
        result, "krippendorff_HH", "krippendorff_H-majority_M-majority"
    )
    result["percentage_agreement_HH_HM_diff"] = _subtract_none_check(
        result, "percentage_agreement_HH", "percentage_agreement_H-majority_M-majority"
    )

    weighted_js, result_bin_details = compute_weighted_js(
        df, "human_labels", "machine_labels", unique_label_list=unique_label_list, bin_type="majority"
    )
    result[f"weighted_JSD_majority"] = weighted_js
    for label in unique_label_list:
        result[f"JSD_H_majority@{label}"] = result_bin_details[label]["jsd"]

    return result


def compute_correlation_categorical(df, unique_label_list):
    result_base = compute_correlation_categorical_num_labeler_agnostic(df, unique_label_list)

    result = {
        "fleiss_HH": compute_fleiss_kappa(df, "human_labels"),
        "fleiss_MM": compute_fleiss_kappa(df, "machine_labels"),
        "fleiss_H-majority_M-majority": compute_fleiss_kappa_2_column(
            df, "human_labels_majority", "machine_labels_majority"
        ),
        "fleiss_H-majority_M-random": compute_fleiss_kappa_2_column(
            df, "human_labels_majority", "machine_random_majority", method="fleiss"
        ),
        "fleiss_HH_HM_diff": None,
        "randolph_HH": compute_fleiss_kappa(df, "human_labels", method="randolph"),
        "randolph_MM": compute_fleiss_kappa(df, "machine_labels", method="randolph"),
        "randolph_H-majority_M-majority": compute_fleiss_kappa_2_column(
            df, "human_labels_majority", "machine_labels_majority", method="randolph"
        ),
        "randolph_H-majority_M-random": compute_fleiss_kappa_2_column(
            df, "human_labels_majority", "machine_random_majority", method="randolph"
        ),
        "randolph_HH_HM_diff": None,
    }
    result["fleiss_HH_HM_diff"] = _subtract_none_check(result, "fleiss_HH", "fleiss_H-majority_M-majority")
    result["randolph_HH_HM_diff"] = _subtract_none_check(result, "randolph_HH", "randolph_H-majority_M-majority")

    result_base.update(result)
    return result_base
