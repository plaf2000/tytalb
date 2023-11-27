import os
import numpy as np
import pandas as pd


def calculate_and_save_stats(data_dict, data_dict_pad, export_dir):
    
    for el, data, file_suffix in zip(["STATISTICS", "STATISTICS PADDED"], [data_dict, data_dict_pad], ["", "_padded"]):
        stats_str = f"-----{el}-----\n\n"
        data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
        labels = list(data.keys())
        counts = list(data.values())

        total_occurrences = sum(counts)
        label_distribution = {label: {'ratio': count/total_occurrences, 'count': count} for label, count in zip(labels, counts)}
        unique_labels = len(labels)

        mean_duration = np.mean(counts) 
        median_duration = np.median(counts)
        std_dev_duration = np.std(counts)
        min_duration = np.min(counts)
        max_duration = np.max(counts)

        stats_str += "Label Distribution:\n"
        for label, values in label_distribution.items():
            stats_str += f"  {label}: {values['ratio']:.2%} ({values['count']} seconds)\n"
        stats_str += f"\nNumber of Unique Labels: {unique_labels}\n\n"
        stats_str += f"Duration Statistics:\n"
        stats_str += f"  Mean Duration: {mean_duration}\n"
        stats_str += f"  Median Duration: {median_duration}\n"
        stats_str += f"  Standard Deviation Duration: {std_dev_duration}\n"
        stats_str += f"  Minimum Duration: {min_duration}\n"
        stats_str += f"  Maximum Duration: {max_duration}\n\n"

        # csv format
        csv_file_path = os.path.join(export_dir, f"stats{file_suffix}.csv")
        pd.DataFrame(label_distribution).to_csv(csv_file_path)
        
        # txt format
        txt_file_path = os.path.join(export_dir, f"stats{file_suffix}.txt")
        with open(txt_file_path, 'a', encoding='utf-8') as txt_file:
            txt_file.write(stats_str)
