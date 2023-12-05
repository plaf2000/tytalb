import os
import numpy as np
import pandas as pd


def calculate_and_save_stats(data_dict, data_dict_pad, export_dir):
    
    for el, data, file_suffix in zip(["STATISTICS", "STATISTICS PADDED"], [data_dict, data_dict_pad], ["", "_padded"]):
        stats_str = f"-----{el}-----\n\n"

        data = dict(sorted(list(map(list, data.items())), key=lambda item: -item[1][1], reverse=False))

        labels = list(data.keys())
        values = list(data.values())

        values_seconds = [el[0] for el in values]
        values_count = [el[1] for el in values]

        sum_values_seconds = sum(values_seconds)
        sum_values_count = sum(values_count)

        label_distribution = {
            label: {
                'ratio_duration': value[0] / sum_values_seconds, 
                'total_duration_[s]': value[0],
                'ratio_count': value[1] / sum_values_count, 
                'annotation_count': int(value[1]),
            } for label, value in zip(labels, values)
        }

        unique_labels = len(labels)

        mean_duration_seconds = np.mean(values_seconds) 
        median_duration_seconds = np.median(values_seconds)
        std_dev_duration_seconds = np.std(values_seconds)
        min_duration_seconds = np.min(values_seconds)
        max_duration_seconds = np.max(values_seconds)

        mean_duration_count = np.mean(values_count) 
        median_duration_count = np.median(values_count)
        std_dev_duration_count = np.std(values_count)
        min_duration_count = np.min(values_count)
        max_duration_count = np.max(values_count)

        stats_str += "Label Distribution:\n"
        stats_str += "The first value indicates the ratio relative to the sum of all labels. The second value tells about the value for a given label\n\n"

        for label, values in label_distribution.items():
            stats_str += f"{label}: \n   Duration: {values['ratio_duration']:.3%}, ({values['total_duration_[s]']:.3f} seconds),\n   Number of occurrences: {values['ratio_count']:.3%}, ({values['annotation_count']})\n"

        stats_str += f"\nNumber of Unique Labels: {unique_labels}\n\n"

        stats_str += f"Duration statistics per label:\n"
        stats_str += f"  Mean: {mean_duration_seconds:.2f} seconds\n"
        stats_str += f"  Median: {median_duration_seconds:.2f} seconds\n"
        stats_str += f"  Standard Deviation: {std_dev_duration_seconds:.2f} seconds\n"
        stats_str += f"  Minimum: {min_duration_seconds:.2f} seconds\n"
        stats_str += f"  Maximum: {max_duration_seconds:.2f} seconds\n\n"

        stats_str += f"Number of occurrences statistics per label:\n"
        stats_str += f"  Mean: {mean_duration_count}\n"
        stats_str += f"  Median: {median_duration_count}\n"
        stats_str += f"  Standard Deviation: {std_dev_duration_count}\n"
        stats_str += f"  Minimum: {min_duration_count}\n"
        stats_str += f"  Maximum: {max_duration_count}\n\n"

        # csv format
        csv_file_path = os.path.join(export_dir, f"stats{file_suffix}.csv")
        df = pd.DataFrame.from_dict(label_distribution, orient="columns").T
        df['annotation_count'] = df['annotation_count'].astype(int)
        df.to_csv(csv_file_path, float_format='%.3f', index_label='label', sep=';')
        
        # txt format
        txt_file_path = os.path.join(export_dir, f"stats{file_suffix}.txt")
        with open(txt_file_path, 'a', encoding='utf-8') as txt_file:
            txt_file.write(stats_str)
