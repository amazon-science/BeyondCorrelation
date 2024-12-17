from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from beyond_correlation.correlation_utils import compute_weighted_js
from beyond_correlation.dataframe_utils import flatten_list


def bar_dist_4(human, human_aggregated, machine, machine_aggregated, unique_label_list, ax):
    all_labels = []
    for label_type, labels in [
        ("$H$", human),
        ("$M$", machine),
        ("$\\overline{H}$", human_aggregated),
        ("$\\overline{M}$", machine_aggregated),
    ]:
        if labels:
            all_labels += [
                {"key_index": i, "key": label, "Probability": Counter(labels)[label] / len(labels), "type": label_type}
                for i, label in enumerate(unique_label_list)
            ]
        else:
            all_labels += [
                {"key_index": i, "key": label, "Probability": 0, "type": label_type}
                for i, label in enumerate(unique_label_list)
            ]
    all_labels = pd.DataFrame(all_labels)

    palette = {"$H$": "red", "$\\overline{H}$": "lightsalmon", "$M$": "green", "$\\overline{M}$": "lightgreen"}
    ax.set_ylim(0, 1.1)
    ax.set_axisbelow(True)
    ax.grid(visible=True, axis="y")
    sns.barplot(all_labels, x="key_index", y="Probability", hue="type", ax=ax, palette=palette, width=0.7)
    ax.set_xlabel(None)
    ax.set_xticks([i for i in range(len(unique_label_list))], [str(x) for x in unique_label_list])
    ax.legend()


def visualize_hm_jsd(
    df,
    unique_label_list,
    human_column_name,
    human_aggregated_column_name,
    machine_column_name,
    machine_aggregated_column_name,
    bin_type="median",
    title="",
    save_loc=None,
    legend=True,
):
    fig = plt.figure(figsize=(3 * len(unique_label_list), 4), tight_layout=True)
    fig_gird = fig.add_gridspec(2, len(unique_label_list), height_ratios=[1, 3])

    weighted_jsd, result_bin_details = compute_weighted_js(
        df, human_column_name, machine_column_name, unique_label_list=unique_label_list, bin_type=bin_type
    )

    proportion_list = [len(result_bin_details[label]["bin_items"]) / len(df) for label in unique_label_list]

    first_bar_ax = None
    for i, label in enumerate(unique_label_list):
        # plot the pie chart on the top
        df_sub = result_bin_details[label]["bin_items"]

        ax_pie = fig.add_subplot(fig_gird[0, i])
        colors = ["lightsteelblue"] * len(unique_label_list)
        colors[i] = "cornflowerblue"
        explode = [0] * len(unique_label_list)
        explode[i] = 0.1
        labels = [""] * len(unique_label_list)
        labels[i] = f"({len(df_sub)} / {len(df)})\n{proportion_list[i] * 100:.1f}%"
        ax_pie.pie(
            proportion_list,
            colors=colors,
            explode=explode,
            labels=labels,
        )

        # create the bottom part of the figure
        if i == 0:
            ax = fig.add_subplot(fig_gird[1, i])
            first_bar_ax = ax
        else:
            ax = fig.add_subplot(fig_gird[1, i], sharex=first_bar_ax, sharey=first_bar_ax)

        ax.set_title(f"$\\overline{{H}}$={label}, JSD: {result_bin_details[label]['jsd'] or np.nan:.2f}")

        bar_dist_4(
            flatten_list(df_sub[human_column_name].tolist()),
            df_sub[human_aggregated_column_name].tolist(),
            flatten_list(df_sub[machine_column_name].tolist()),
            df_sub[machine_aggregated_column_name].tolist(),
            ax=ax,
            unique_label_list=unique_label_list,
        )
        if i > 0:
            ax.set_yticklabels([])
            ax.set_ylabel(None)

        if legend:
            if i < len(unique_label_list) - 1:
                ax.legend().remove()
        else:
            ax.legend().remove()

    title = (title + f"\nbin_type: {bin_type}; weighted_JSD: {weighted_jsd:.2f}").strip()
    fig.suptitle(title)
    if save_loc:
        fig.savefig(save_loc, bbox_inches="tight")

    return fig
