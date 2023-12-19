import csv
from dataclasses import dataclass
import warnings
from copy import deepcopy
from functools import cached_property
from audio_file import AudioFile
from segment import Segment
import json
import fnmatch
import re
from loggers import Logger
from parsers import get_parser
from loggers import ProcLogger, Logger, ProgressBar 
from stats import calculate_and_save_stats
from variables import NOISE_LABEL, AUDIO_EXTENSION_PRIORITY, BIRDNET_AUDIO_DURATION
import pandas as pd
import numpy as np
import os
from units import TimeUnit
from collections import defaultdict


class LabelMapper:
    def __init__(self, label_settings_path: str, *args, **kwargs):

        try:
            with open(label_settings_path, encoding='utf-8') as fp:
                self.json_obj = json.load(fp)
        except:
            self.json_obj = {}
        self.map_dict: dict = self.json_obj.setdefault("map", {})
        self.sub_dict: dict = self.json_obj.setdefault("sub", {})
        self.whitelist = self.json_obj.setdefault("whitelist", None)
        self.blacklist = self.json_obj.setdefault("blacklist", None)
        self.strip = self.json_obj.setdefault("strip", True)
        self.single_ws = self.json_obj.setdefault("single whitespace", True)
        self.lowercase = self.json_obj.setdefault("lower", False)
    
    def black_listed(self, label: str) -> bool:
        if self.whitelist:
            return label not in self.whitelist
        return label in self.blacklist if self.blacklist else False
    
    def clean(self, label: str) -> str:
        if self.strip:
            label = label.strip()
        if self.single_ws:
            label = " ".join(label.split())
        if self.lowercase:
            label = label.lower()
        return label
    
    def map(self, label: str) -> str:
        for k, v in self.map_dict.items():
            if re.match(k, label):
                label = v
                break
        if self.black_listed(label):
            label = "Noise"
        return label
    
    def sub(self, label: str) -> str:
        for k, v in self.sub_dict.items():
            label = re.sub(k, v, label)
        return label
    
    
    def do_all(self, label: str) -> str:
        return self.map(self.sub(self.clean(label)))

    
    
class SegmentsWrapper:
    def __init__(self, unique = True, audio_file: AudioFile | None = None, segments: list[Segment] | None  = None):
        if segments is None:
            segments = []
        self.unique = unique
        self.segments: list[Segment] = segments
        self.audio_file: AudioFile | None = audio_file
    
    
    @property
    def segments_interval_tree(self):
        if self.segments:
            return self.segments[0].get_intervaltree(self.segments)
        else:
            return Segment.get_intervaltree(self.segments)
    
        

class Annotations:
    def __init__(
            self,
            tables_dir: str,
            table_format: str,
            recursive_subfolders = True,
            **parser_kwargs
        ):

        self.parser = get_parser(table_format, **parser_kwargs)
        self.tables_dir = os.path.normpath(tables_dir)
        self.recursive_subfolders = recursive_subfolders
        self.tables_paths: list = []
        self.audio_files: None | dict[str, SegmentsWrapper] = None

        if recursive_subfolders:
            for dirpath, dirs, files in os.walk(self.tables_dir):
                for filename in files:
                    if not self.parser.is_table(filename):
                        continue

                    fpath = os.path.join(dirpath, filename)
                    self.tables_paths.append(fpath)
        else:
            for f in os.listdir(self.tables_dir):
                fpath = os.path.join(self.tables_dir, f)
                if os.path.isfile(fpath) and self.parser.is_table(fpath):
                    self.tables_paths.append(fpath)

        if len(self.tables_paths) == 0:
            raise AttributeError("No annotations found in the provided folder.")

    def load(self, logger: Logger = Logger(log=False), list_rel_paths: list[str] = None, **kwargs):

        self.audio_files: dict[str, SegmentsWrapper] = dict()
        prog_bar = ProgressBar("Reading tables", len(self.tables_paths))
               
                
        self.annotation_label_linenumber: list[tuple[str, int, str]] = []
        

        for table_path in self.tables_paths:
            if self.parser.is_table_per_file(table_path)\
               and list_rel_paths is not None\
               and self.parser.get_audio_rel_no_ext_path(table_path, self.tables_dir) not in list_rel_paths:                
                
                continue
            for rel_path, segment in zip(self.parser.get_audio_rel_no_ext_paths(table_path, self.tables_dir), 
                                         self.parser.get_segments(table_path,logger=logger)):
                if list_rel_paths is not None and rel_path not in list_rel_paths:
                    continue
                path_safe_label = ' '.join(re.sub(r'[\\/*?:"<>|]', '', segment.label).split())


                self.annotation_label_linenumber.append((rel_path, segment.line_number, segment.label))

                basename = os.path.basename(rel_path)
                if rel_path in self.audio_files.keys():
                    self.audio_files[rel_path].segments.append(segment)
                    continue
                    
                unique = True
                for rp in self.audio_files.keys():
                    bnm = os.path.basename(rp)
                    if basename == bnm:
                        logger.print(f"Found two files with the same name: {basename} ({rp} and {rel_path}). "
                                      "The output files will contain the path to ensure uniquess.")
                        unique=False
                        break
                self.audio_files.setdefault(rel_path, SegmentsWrapper(unique)).segments.append(segment)
            prog_bar.print(1)
                
        for v in self.audio_files.values():
            v.segments = sorted(v.segments, key=lambda seg: seg.tstart)
        prog_bar.terminate()

    def map_labels(self, label_settings_path=None, **kwargs):
        if label_settings_path is not None and os.path.isfile(label_settings_path):
            prog_bar = ProgressBar("Changing labels", self.n_segments)
            label_mapper = LabelMapper(label_settings_path)
            for af_wrap in self.audio_files.values():
                for seg in af_wrap.segments:
                    seg.label = label_mapper.do_all(seg.label)
                    prog_bar.print(1)
            prog_bar.terminate()
    
    def correct_labels(self, out_dir: str, label_settings_path=None,  **kwargs):
        if label_settings_path is not None and os.path.isfile(label_settings_path):
            prog_bar = ProgressBar("Correcting labels", len(self.tables_paths))
            label_mapper = LabelMapper(label_settings_path)
            for table_path in self.tables_paths:
                out_path = os.path.join(out_dir, os.path.relpath(table_path, self.tables_dir))
                self.parser.edit_label(table_path, label_mapper, new_table_path=out_path)
                prog_bar.print(1)
            prog_bar.terminate()
    
    def load_audio_paths(self, audio_files_dir: str, **kwargs):
        prog_bar = ProgressBar("Retrieving audio paths", self.n_segments)
        for rel_path, af_wrap in self.audio_files.items():
            rel_parent = os.path.dirname(rel_path)
            parent = os.path.join(audio_files_dir, rel_parent)
            audiodir_files = [os.path.join(rel_parent, os.path.basename(f)) for f in os.listdir(parent) if os.path.isfile(os.path.join(parent, f))
                            and f.split(".")[-1].lower() in AUDIO_EXTENSION_PRIORITY]
            # Get all files starting with this filename
            audio_candidates = fnmatch.filter(audiodir_files, f"{rel_path}.*")

            if not audio_candidates:
                raise Exception(f"No audio files found starting with relative path "\
                                f"{rel_path} and extension {'|'.join(AUDIO_EXTENSION_PRIORITY)} "\
                                f"inside {audio_files_dir}.")
            # Give the priority based on `AUDIO_EXTENSION_PRIORITY`
            priority = lambda fname: AUDIO_EXTENSION_PRIORITY.index(fname.split(".")[-1].lower())
            chosen_audio_rel_path = min(audio_candidates, key = priority)

            audio_path = os.path.join(audio_files_dir, chosen_audio_rel_path)
            af = AudioFile(audio_path)
            af.rel_path = chosen_audio_rel_path
            af.set_date(**kwargs)
       
            af_wrap.audio_file = af
            prog_bar.print(1)
        prog_bar.terminate()
        

    @property
    def n_tables(self):
        return len(self.tables_paths)
    
    @cached_property
    def n_segments(self):
        return sum([len(af_wrap.segments) for af_wrap in self.audio_files.values()])

    def extract_for_training(self, audio_files_dir: str, export_dir: str, logger: Logger, include_path=False, stats_only=False, occurrence_threshold=0, **kwargs):
        """
        Extract BIRDNET_AUDIO_DURATION-long chunks to train a custom classifier.
        """
        logger.print("Input annotations' folder:", self.tables_dir)
        logger.print("Input audio folder:", audio_files_dir)
        logger.print("Output audio folder:", export_dir)


        self.load(logger, **kwargs)
        self.map_labels(**kwargs)

        annotation_label_linenumber_file = os.path.join(export_dir, "labels_filenames_linenumber.csv")
       

        df = pd.DataFrame(self.annotation_label_linenumber, columns=["File name", "Line number", "Label"])
        df["Occurrences"] = 1
        df_occur = df.groupby("Label").sum()["Occurrences"]
        for label in df_occur.index:
            df.loc[df["Label"] == label, "Occurrences"] = df_occur[label]
        df = df.sort_values(by=["Occurrences","Label", "File name", "Line number"])\
               .reset_index(drop=True)
        df.index = df.index.rename("Annotation id")
        df.to_csv(annotation_label_linenumber_file)        

        if not stats_only:
            self.load_audio_paths(audio_files_dir, **kwargs)

        prog_bar = ProgressBar("Generating statistics" if stats_only else "Exporting segments", self.n_segments)
        proc_logger = ProcLogger(**kwargs)
        logger.print("Found", len(self.audio_files), "audio files.")

        stats = defaultdict(lambda: [0, 0])
        stats_pad = defaultdict(lambda: [0, 0])

        for af_wrap in self.audio_files.values():
            if include_path or not af_wrap.unique:
                # If the filename is not unique (or the user decides to) include  
                # the relative path in the output filename.
                af = af_wrap.audio_file
                path = os.path.normpath(os.path.dirname(af.rel_path))
                splits = path.split(os.sep)
                if path!=".":
                    af.prefix = f"{'_'.join(splits)}_{af.prefix}"    
            for segment in af_wrap.segments:
                segment_pad = segment.birdnet_pad()
                label = segment.label
                stats[label][0] += segment.dur
                stats[label][1] += 1

                stats_pad[label][0] += segment_pad.dur
                stats_pad[label][1] += 1
                
            if not stats_only:
                af_wrap.audio_file.export_all_birdnet(export_dir, af_wrap.segments, proc_logger=proc_logger, logger=logger, progress_bar=prog_bar, **kwargs)

        calculate_and_save_stats(stats, stats_pad, export_dir)
        prog_bar.terminate()

    def filter_confidence(self, confidence_threshold: float):
        """
        Returns an `Annotation` deepcopy object where all segments below the
        confidence threshold are removed.
        """

        copy = deepcopy(self)
        if self.audio_files is None:
            self.load()
        for rel_path in copy.audio_files.keys():
            segs = copy.audio_files[rel_path].segments
            copy.audio_files[rel_path].segments = [s for s in segs if s.confidence >= confidence_threshold]
        return copy


    def validate(self, other: 'Annotations', *args, **kwargs):
        validate(ground_truth=self, to_validate=other, *args, **kwargs) 
    

def validate(
        ground_truth: Annotations,
        to_validate: Annotations,
        filter_confidence: float = None,
        binary=False,
        positive_labels: str=None,
        overlapping_threshold_s: float = .5,
        late_start = False,
        early_stop = False,
        skip_missing_gt = True,
        logger: Logger = Logger(log=False),
        single_row = False,
        **kwargs
        ):
    """
    Compare the annotations to validate to the ground truth ones.
    Returns a tuple of shape (2,2). The first element of the tuple
    contains the confusion matrix and the metrics (as tuple) 
    in terms of the count of overlapping segments, while the second 
    in terms of the overlapping time.

    Problems
    --------
     - If ground-truth annotation times are overlapping, in the 
       matrix they will appear multiple times, so some annotations
       can be considered wrong although correct.
     - The true negatives (noise matching noise) are not registered
       in the matrix.
    """

    ground_truth.load()
    if "gt_label_settings_path" in kwargs:
        ground_truth.map_labels(kwargs["gt_label_settings_path"])


    # Only load the tables that we want to validate (saves time)
    if skip_missing_gt:
        gt_paths = ground_truth.audio_files.keys()
        to_validate.load(list_rel_paths=gt_paths)
        if filter_confidence is not None:
            to_validate = to_validate.filter_confidence(filter_confidence)
        
    if not ground_truth.audio_files:
        raise ValueError("No ground truth annotations found.")

    all_rel_paths = set(ground_truth.audio_files.keys()) | set(to_validate.audio_files.keys())
    labels: set[str] = set()
    overlapping_threshold = TimeUnit(overlapping_threshold_s)


    # Union the labels in ground truth and set to validate
    for bnt in [ground_truth, to_validate]:
        for af_wrapper in bnt.audio_files.values():
            for seg in af_wrapper.segments:
                labels.add(seg.label)
    
    labels.add(NOISE_LABEL)
    labels = sorted(labels)
    n_labels = len(labels)

    if binary and n_labels>2:
        ground_truth = deepcopy(ground_truth)
        to_validate = deepcopy(to_validate)
        if positive_labels is None:
            raise AttributeError("Binary classification with more than one label! Please specify the positive label(s).")
        if isinstance(positive_labels, str):
            positive_labels = [positive_labels]
        found = False
        for bnt in [ground_truth, to_validate]:
            for af_wrapper in bnt.audio_files.values():
                for seg in af_wrapper.segments:
                    # Label the positive segments as "positive" and the other with `NOISE_LABEL`
                    if seg.label not in positive_labels:
                        seg.label = NOISE_LABEL
                    else:
                        seg.label = "Positive"
                        found = True
        if not found:
            warnings.warn(f"No label from \"{','.join(positive_labels)}\" found.")
        labels = ["Positive", NOISE_LABEL]
        n_labels = len(labels)          
    
    index: dict[str, int] = {}

    # Map label to the row/column of the confusion matrix
    for i, label in enumerate(labels):
        index[label] = i
    
    # Confusion matrices for time and count
    conf_time_matrix = np.zeros((n_labels, n_labels), np.float64)
    conf_count_matrix = np.zeros((n_labels, n_labels), np.int64)

    # Shortcuts for setting the confusion matrices
    def set_conf_time(label_truth, label_prediction, duration):
        conf_time_matrix[index[label_truth], index[label_prediction]] += duration
    
    def set_both(label_truth, label_prediction, duration):
        set_conf_time(label_truth, label_prediction, duration)
        # If there is some overlap, we add one to the count
        if duration >= overlapping_threshold:
            conf_count_matrix[index[label_truth], index[label_prediction]] += 1


    for rel_path in all_rel_paths:
        af_gt, af_tv = None, None
        # Get the audio file for the ground truth
        if rel_path in ground_truth.audio_files:
            af_gt = ground_truth.audio_files[rel_path]

        # Get the audio file for the validation
        if rel_path in to_validate.audio_files:
            af_tv = to_validate.audio_files[rel_path]

        # If there are no labels in the ground truth, there are only FP
        # Unless skip_missing_gt is True
        if not rel_path in ground_truth.audio_files:
            warnings.warn(f"{rel_path} not found in ground truth annotations.")
            for seg in af_tv.segments:
                if seg.label != NOISE_LABEL:
                    set_both(NOISE_LABEL, seg.label, seg.dur)
            continue
        
        # If there are no labels in the validation, there are only FN
        if not rel_path in to_validate.audio_files:
            warnings.warn(f"{rel_path} not found in annotations to validate.")

            for seg in af_gt.segments:
                if seg.label != NOISE_LABEL:
                    set_both(seg.label, NOISE_LABEL, seg.dur)
            continue

        min_tstart = min([s.tstart for s in af_gt.segments])
        max_tend = max([s.tend for s in af_gt.segments])

        def interval_tree(af: SegmentsWrapper):
            # If late_start and early_stop, restrict the interval
            return Segment.get_intervaltree([s for s in af.segments if s.label != NOISE_LABEL 
                                             and (not late_start or s.tend >= min_tstart)
                                             and (not early_stop or s.tstart <= max_tend)])
        
        segs_gt = interval_tree(af_gt)
        segs_tv = interval_tree(af_tv)
    

        for seg_gt in segs_gt:
            seg_gt = Segment.from_interval(seg_gt)

            overlapping = segs_tv[seg_gt.begin:seg_gt.end]

            if len(overlapping) == 0:
                # The ground truth has no label for this interval, therefore FP
                set_both(seg_gt.label, NOISE_LABEL, seg_gt.dur)
                continue

            for seg_tv in overlapping:
                # If overlapping set the confusion matrices accordingly
                seg_tv = Segment.from_interval(seg_tv)
                set_both(seg_gt.label, seg_tv.label, seg_tv.overlapping_time(seg_gt))
            
            # Set all the time in which no overlap was found as FP
            t = Segment.get_intervaltree(overlapping)
            t.merge_overlaps()
            tot_overlapping = sum([Segment.from_interval(s).overlapping_time(seg_gt) for s in t])
            set_conf_time(seg_gt.label, NOISE_LABEL, seg_gt.dur - tot_overlapping)

        for seg_tv in segs_tv:
            seg_tv = Segment.from_interval(seg_tv)
            overlapping = segs_gt[seg_tv.begin: seg_tv.end]
            if len(overlapping) == 0:
                # The annotation to validate have no label for this interval, therefore FN
                set_both(NOISE_LABEL, seg_tv.label, seg_gt.dur)
                continue

            # Set all the time in which no overlap was found as FP
            t = Segment.get_intervaltree(overlapping)
            t.merge_overlaps()
            tot_overlapping = sum([Segment.from_interval(s).overlapping_time(seg_gt) for s in t])
            set_conf_time(NOISE_LABEL, seg_tv.label, seg_tv.dur - tot_overlapping)

    
    def stats(matrix, f_scores_betas = [.5, 1, 2]):
        precision = {}
        recall = {}
        true_positive = {}
        false_positive = {}
        false_negative = {}

        def fbeta_score(beta: float, p: float, r: float):
            return 0 if p + r == 0 else (1 + beta**2) * (p * r) / ((p * beta**2) + r)
        

        precscore = lambda tp, fp: 0 if tp==0 else tp / (tp + fp)
        recallscore = lambda tp, fn: 0 if tp==0 else tp / (tp + fn)

        data = {}
        
        f_scores = {b: {} for b in f_scores_betas}

        for i, label in enumerate(labels):
            if label == NOISE_LABEL and (binary or n_labels==2):
                continue
            mask = np.ones_like(matrix[i], np.bool_)
            mask[i] = 0
            true_positive[label] = tp = matrix[i,i]
            false_positive[label] = fp = np.dot(matrix[:, i], mask)
            false_negative[label] = fn = np.dot(matrix[i, :], mask)
            if label == NOISE_LABEL:
                # Skip because TP noise are unknown
                continue
            precision[label] = p = precscore(tp, fp)
            recall[label] = r = recallscore(tp, fn)
            for b in f_scores_betas:
                f_scores[b][label] = fbeta_score(b, p, r)


        macro_average_label = "Macro-average"
        micro_average_label = "Micro-average"
        
        non_noise_values = lambda dict: [dict[k] for k in dict.keys() if k != NOISE_LABEL 
                                                                         and k != macro_average_label 
                                                                         and k != micro_average_label]

        # Macro-average:
        false_positive[macro_average_label] = pd.NA
        false_negative[macro_average_label] = pd.NA
        precision[macro_average_label] = p = np.mean(non_noise_values(precision))
        recall[macro_average_label] = r = np.mean(non_noise_values(recall))
        for b in f_scores_betas:
            f_scores[b][macro_average_label] = fbeta_score(b, p, r)

         # Micro-average:
        false_positive[micro_average_label] = fp = np.sum(non_noise_values(false_positive))
        false_negative[micro_average_label] = fn = np.sum(non_noise_values(false_negative))
        tp = np.sum(list(true_positive.values()))
        precision[micro_average_label] = p =  precscore(tp, fp)
        recall[micro_average_label] = r = recallscore(tp, fn)
        for b in f_scores_betas:
            f_scores[b][micro_average_label] = fbeta_score(b, p, r)


        df_matrix = pd.DataFrame(data=matrix, index=labels, columns=labels)
        # TODO: TN (TP noise) is never set. Maybe should be computed, but not so relevant.
        if df_matrix.loc[NOISE_LABEL, NOISE_LABEL].dtype == np.int64:
            df_matrix[NOISE_LABEL] = df_matrix[NOISE_LABEL].astype("Int64")
        df_matrix.loc[NOISE_LABEL, NOISE_LABEL] = pd.NA

        df_matrix.index.name = r"True\Prediction"


        data = {
            "precision": precision,
            "recall": recall,
        }
        data |= { f"f{b} score":  f_scores[b] for b in f_scores_betas }
        data |= {
            "false positive": false_positive,
            "false negative": false_negative,
        }


        if single_row:
            new_data = dict()
            for label in labels + [macro_average_label, micro_average_label]:
                for metric_name, metric_values in data.items():
                    if label in metric_values:
                        new_data[f"{label} {metric_name}"] = [metric_values[label]]
            data = new_data

        df_metrics =  pd.DataFrame(
            data,
        )

        return df_matrix, df_metrics

    return stats(conf_time_matrix), stats(conf_count_matrix)

def plot_stats(stats: tuple[pd.DataFrame, pd.DataFrame], fout: str):
    # TODO: Implement this
    conf, metrics = stats   
    labels = conf.columns.drop(["Confidence"])


def plot_both_stats(statss, fouts):
    for stats, fout in zip(statss, fouts):
        plot_stats(stats, fout)