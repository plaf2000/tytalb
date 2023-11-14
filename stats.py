import os
import numpy as np


def calculate_and_save_stats(data_dict, export_dir):
    labels = list(data_dict.keys())
    counts = list(data_dict.values())

    total_occurrences = sum(counts)
    label_distribution = {label: {'ratio': count/total_occurrences, 'count': count} for label, count in zip(labels, counts)}
    unique_labels = len(labels)

    mean_duration = np.mean(counts) 
    median_duration = np.median(counts)
    std_dev_duration = np.std(counts)
    min_duration = np.min(counts)
    max_duration = np.max(counts)

    stats_str = "Label Distribution:\n"
    for label, values in label_distribution.items():
        stats_str += f"  {label}: {values['ratio']:.2%} ({values['count']} seconds)\n"
    stats_str += f"\nNumber of Unique Labels: {unique_labels}\n\n"
    stats_str += f"Duration Statistics:\n"
    stats_str += f"  Mean Duration: {mean_duration}\n"
    stats_str += f"  Median Duration: {median_duration}\n"
    stats_str += f"  Standard Deviation Duration: {std_dev_duration}\n"
    stats_str += f"  Minimum Duration: {min_duration}\n"
    stats_str += f"  Maximum Duration: {max_duration}\n"

    stats_file_path = os.path.join(export_dir, "stats.txt")
    with open(stats_file_path, 'w') as stats_file:
        stats_file.write(stats_str)
        