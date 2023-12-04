import subprocess
from annotations import Annotations, validate
from parsers import ap_names
from loggers import Logger 
from variables import BIRDNET_AUDIO_DURATION, BIRDNET_SAMPLE_RATE
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
from calls_finder import find_calls, multi_processes
import pandas as pd
import numpy as np
import time
import os




def default_label_settings(label_settings_path, tables_dir):
    if label_settings_path is None:
        label_settings_path = os.path.join(tables_dir, "labels.json")
    return label_settings_path
    




if __name__ == "__main__":
    ts = time.time()
    arg_parser = ArgumentParser()
    arg_parser.description = f"Train and validate a custom BirdNet classifier based on given annotations by first exporting "\
                             f"{BIRDNET_AUDIO_DURATION.s}s segments."
    
    subparsers = arg_parser.add_subparsers(dest="action")

    """
        Parse arguments to extract audio chunks.
    """

    extract_parser = subparsers.add_parser("extract",
                                           help="Extracts audio chunks from long audio files using FFmpeg based on the given parser annotation. " \
                                                'The result consists of multiple audio files ("chunks"), each 3s long, placed in the corresponding ' \
                                                "labelled folder, which can be used to train the BirdNet custom classifier.")
    
    extract_parser.add_argument("-i", "--input-dir",
                                dest="tables_dir",
                                help="Path to the folder of the (manual) annotations.",
                                default=".")
    
    extract_parser.add_argument("-re", "--recursive",
                                type=bool,
                                dest="recursive",
                                help="Wether to look for tables inside the root directory recursively or not (default=True).",
                                default=True,
                                action=BooleanOptionalAction)
    
    extract_parser.add_argument("-f", "--annotation-format",
                                dest="table_format",
                                choices=ap_names,
                                required=True,
                                help="Annotation format.")
    
    extract_parser.add_argument("--header",
                                dest="header",
                                help="Whether the annotation tables have an header. The default value is defined "\
                                     "by the annotations parser.",
                                action=BooleanOptionalAction)
    
    
    extract_parser.add_argument("-a", "--audio-root-dir",
                                dest="audio_files_dir",
                                help="Path to the root directory of the audio files (default=current working dir).", default=".")
    
    extract_parser.add_argument("-oe", "--audio-output-ext",
                                dest="audio_output_ext",
                                help="Key-sensitive extension of the output audio files (default=flac).", default="flac")
    

    extract_parser.add_argument("-o", "--output-dir",
                                dest="export_dir",
                                help="Path to the output directory. If doesn't exist, it will be created.",
                                default=".")

    extract_parser.add_argument("--tstamp-subdir",
                                dest="tstamp_outdir",
                                help="Whether to create an output subfolder with the current timestamp which "\
                                     "which will contain the output files.",
                                default=True,
                                action=BooleanOptionalAction)

    extract_parser.add_argument("-l", "--label-settings",
                                dest="label_settings_path",
                                help="Path to the file used to map and filter labels. Please refer to `README.md`. "\
                                     "By default the file is `labels.json` in the root directory of annotations.",
                                type=str,
                                default=None)
    
    extract_parser.add_argument("-r", "--resample",
                                dest="resample",
                                help=f"Resample the chunk to the given value in Hz. (default={BIRDNET_SAMPLE_RATE})",
                                type=int,
                                default=BIRDNET_SAMPLE_RATE)
    
    extract_parser.add_argument("-co", "--chunk-overlap",
                                dest="chunk_overlap",
                                help=f"Overlap in seconds between chunks for segments longer than {BIRDNET_AUDIO_DURATION.s}s. "\
                                     F"If it is 0 (by default) the program may run faster.",
                                default=0)
    
    extract_parser.add_argument("-df", "--date-format",
                                dest="date_format",
                                help='Date format of the file. (default = "%%Y%%m%%d_%%H%%M%%S")',
                                type=str,
                                default="%Y%m%d_%H%M%S")
    
    extract_parser.add_argument("-ls", "--late-start",
                                dest="late_start",
                                help='Whether to not consider the interval between the start of the recording and the first '\
                                     'annotation (default = False)',
                                type=bool,
                                action=BooleanOptionalAction,
                                default=False)
    
    extract_parser.add_argument("-es", "--early-stop",
                                dest="early_stop",
                                help='Whether to not consider the interval between the last annotation '\
                                     'and the end of the recording (default = False)',
                                type=bool,
                                action=BooleanOptionalAction,
                                default=False)
    
    extract_parser.add_argument("-ip", "--include-path",
                                dest="include_path",
                                help='Whether to include the relative path in the output file name (default = False). '\
                                     'If two filenames are not unique, this will be done automatically.',
                                type=bool,
                                action=BooleanOptionalAction,
                                default=False)
    
    extract_parser.add_argument("-so", "--stats-only",
                                dest="stats_only",
                                help='Whether to calculate only input data statistics',
                                type=bool,
                                action=BooleanOptionalAction,
                                default=False)
    
    extract_parser.add_argument("-nr", "--noise-ratio",
                                dest="noise_export_ratio",
                                help="Ratio between 0 and 1 (in terms of track's length) of noise to export (default = 0.1).",
                                type=float,
                                default=.1)
    
    # extract_parser.add_argument("-d", "--debug",
    #                             dest="debug",
    #                             help='Whether to log debug informations too.',
    #                             type=bool,
    #                             action=BooleanOptionalAction,
    #                             default=False)

    
    """
        Parse arguments to train the model.
    """
    train_parser = subparsers.add_parser("train", help="Train a custom classifier using BirdNet Analyzer. "\
                                                       "The args are passed directly to `train.py` from BirdNet.")



    """
        Parse arguments to validate BirdNet predictions.
    """
    validate_parser = subparsers.add_parser("validate", help="Validate the output from BirdNet Analyzer with some ground truth annotations. "\
                                                             "This creates two confusion matrices: one for the time (`confusion_matrix_time.csv`) "\
                                                             "and one for the count (`confusion_matrix_count.csv`)"\
                                                             "of (in)correctly identified segments of audio. From this, recall, precision and "\
                                                             "f1 score are computed and output in different tables (`validation_metrics_count.csv` "\
                                                             "and `validation_metrics_time.csv`).")

    validate_parser.add_argument("-gt", "--ground-truth",
                                dest="tables_dir_gt",
                                help="Path to the folder of the ground truth annotations (default=current working dir).",
                                default=".")
    
    validate_parser.add_argument("-tv", "--to-validate",
                            dest="tables_dir_tv",
                            help="Path to the folder of the annotations to validate (default=current working dir).",
                            default=".")
    
    validate_parser.add_argument("-fgt", "--annotation-format-ground-truth",
                                dest="table_format_gt",
                                required=True,
                                choices=ap_names,
                                help="Annotation format for ground truth data.")

    validate_parser.add_argument("-ftv", "--annotation-format-to-validate",
                                dest="table_format_tv",
                                choices=ap_names,
                                help="Annotation format for data to validate (default=raven).",
                                default="birdnet_raven")
    
    validate_parser.add_argument("-o", "--output-dir",
                                dest="output_dir",
                                help="Path to the output directory (default=current working dir).",
                                default=".")
    
    validate_parser.add_argument("-re", "--recursive",
                                dest="recursive",
                                help="Wether to look for tables inside the root directory recursively or not (default=True).",
                                type=bool,
                                action=BooleanOptionalAction,
                                default=True)
    
    validate_parser.add_argument("-lgt", "--label-settings-gt",
                                dest="gt_label_settings_path",
                                help="Path to the file used to map and filter labels of the ground truth annotations."\
                                     "Please refer to `README.md`. By default the file is `labels.json` in the root"\
                                     " directory of annotations.",
                                type=str,
                                default=None)

    validate_parser.add_argument("-ls", "--late-start",
                                dest="late_start",
                                help='Whether to not consider the interval between the start of the ground truth recording '\
                                     'and the first annotation (default = False)',
                                type=bool,
                                action=BooleanOptionalAction,
                                default=False)
    
    validate_parser.add_argument("-es", "--early-stop",
                                dest="early_stop",
                                help='Whether to not consider the interval between the last annotation '\
                                     'and the end of the recording (default = False)',
                                type=bool,
                                action=BooleanOptionalAction,
                                default=False)
    
    validate_parser.add_argument("-b", "--binary",
                                dest="binary",
                                help='Whether to validate as binary classification. If set, and '\
                                     'the POSITIVE_LABEL is not provided, an exception will be raised.',
                                type=bool,
                                action=BooleanOptionalAction,
                                default=False)
    
    validate_parser.add_argument("-p", "--positive-labels",
                                dest="positive_labels",
                                help='Comma-separated labels considered as positive for the binary classification.',
                                type=str)
    
    validate_parser.add_argument("-cts", "--conf-thresholds-start",
                                 dest="confidence_thresholds_start",
                                 help="Start range for confidence thresholds.",
                                 type=float,
                                 default=0)
    
    validate_parser.add_argument("-cte", "--conf-thresholds-end",
                                 dest="confidence_thresholds_end",
                                 help="End range for confidence thresholds",
                                 type=float,
                                 default=1)
    
    validate_parser.add_argument("-ct", "--conf-thresholds",
                                 dest="confidence_thresholds",
                                 help="Number of thresholds to filter the data to validate "\
                                      "(linearly distributed between CONFIDENCE_THRESHOLDS_START and "\
                                      "CONFIDENCE_THRESHOLDS_END). The table format must have a field "\
                                      "for the confidence and it has to be defined in the parser.",
                                 type=int,
                                 default=None)
    
    validate_parser.add_argument("-ot", "--overlapping-threshold",
                                 dest="overlapping_threshold_s",
                                 help="Overlap threshold in seconds between two segments to consider them (correctly) classified."\
                                      "(default = 0.5).",
                                 type=float,
                                 default=.5)
    
    validate_parser.add_argument("-smgt", "--skip-missing-gt",
                                 dest="skip_missing_gt",
                                 help="Whether to skip the missing ground-truth file or consider them as noise."\
                                      "(default = True).",
                                type=bool,
                                action=BooleanOptionalAction,
                                default=True)

    """
        Parse arguments to find single calls.
    """
    calls_parser = subparsers.add_parser("calls", help="Given the birdnet output, find the single calls")


    calls_parser.add_argument("-p", "--processes",
                                dest="n_processes",
                                help=f"Number of processes. Each processs operate on different files. (default={os.cpu_count()})",
                                type=int,
                                default=os.cpu_count())
    
    calls_parser.add_argument("-a", "--audio-root-dir",
                                dest="audio_files_dir",
                                help="Path to the root directory of the audio files (default=current working dir).", default=".")
    
    calls_parser.add_argument("-i", "--input-dir",
                                dest="tables_dir",
                                help="Annotations' directory given by BirdNET's analysis (default=current working dir).",
                                default=".")

    calls_parser.add_argument("-o", "--output-dir",
                                dest="output_dir",
                                help=f"Directory where to output the new annotations.",
                                default=".")
    
    calls_parser.add_argument("-c", "--custom-classifier",
                                dest="custom_classifier",
                                help=f"Path to the custom classifier to identify the calls.")
    
    args, custom_args = arg_parser.parse_known_args()


    if args.action == "extract":
        os.makedirs(args.export_dir, exist_ok=True)
        export_dir = args.export_dir       
        if args.tstamp_outdir:
            export_dir = os.path.join(args.export_dir, datetime.utcnow().strftime("%Y%m%d_%H%M%SUTC"))
            os.mkdir(export_dir)
        

        logger = Logger(logfile_path=os.path.join(export_dir, "log.txt"))

        logger.print(args)

        logger.print("Started processing...")
        ts = time.time()



        label_settings_path = default_label_settings(args.label_settings_path, args.tables_dir)

        parser_kwargs = {}
        if args.header is not None:
            parser_kwargs["header"] = args.header


        try:
            bnt = Annotations(
                tables_dir=args.tables_dir,
                table_format=args.table_format,
                recursive_subfolders=args.recursive,
                logger=logger,
                **parser_kwargs,
            )\
            .extract_for_training(
                audio_files_dir=args.audio_files_dir,
                export_dir=export_dir,
                audio_format=args.audio_output_ext,
                logger=logger,
                logfile_errors_path=os.path.join(export_dir, "error_log.txt"),
                logfile_success_path=os.path.join(export_dir, "success_log.txt"),
                label_settings_path = label_settings_path,
                resample=args.resample,
                overlap_s=args.chunk_overlap,
                date_format=args.date_format,
                late_start = args.late_start,
                early_stop = args.early_stop,
                include_path = args.include_path,
                stats_only = args.stats_only,
                noise_export_ratio = args.noise_export_ratio,
            )
        
            logger.print(f"... end of processing (elapsed {time.time()-ts:.1f}s)")
        except Exception as e:
            print()
            print("An error occured and the operation was not completed!")
            print(f"Check {logger.logfile_path} for more information.")
            logger.print_exception(e)  


    elif args.action == "train":
        subprocess.run(["python", "train.py"] + custom_args, cwd="BirdNET-Analyzer/")
    elif args.action=="validate":

        bnt_gt = Annotations(
            tables_dir=args.tables_dir_gt,
            table_format=args.table_format_gt,
            logger=Logger(),
            recursive_subfolders=args.recursive
        )

        bnt_tv = Annotations(
            tables_dir=args.tables_dir_tv,
            table_format=args.table_format_tv,
            logger=Logger(),
            recursive_subfolders=args.recursive
        )

        positive_labels = None
        if args.positive_labels is not None:
            positive_labels = args.positive_labels.split(",")

        gt_label_settings_path = default_label_settings(args.gt_label_settings_path, args.tables_dir_gt)
        

        if args.confidence_thresholds is not None:
            thresholds = np.linspace(args.confidence_thresholds_start, args.confidence_thresholds_end, args.confidence_thresholds)
            stats_time: list[pd.DataFrame, pd.DataFrame] = [pd.DataFrame(), pd.DataFrame()]
            stats_count: list[pd.DataFrame, pd.DataFrame] = [pd.DataFrame(), pd.DataFrame()]
            for t in thresholds:
                t = round(t, 4)
                stime, scount = validate(
                    ground_truth = bnt_gt,
                    to_validate = bnt_tv,
                    filter_confidence = t,
                    binary = args.binary,
                    positive_labels = positive_labels,
                    late_start = args.late_start,
                    early_stop = args.early_stop,
                    overlapping_threshold_s = args.overlapping_threshold_s,
                    skip_missing_gt=args.skip_missing_gt,
                    gt_label_settings_path = gt_label_settings_path
                )

                for s in [stime, scount]:
                    for df in s:
                        df["Confidence threshold"] = t
                        
                for i, df in enumerate(stime):
                    stats_time[i] = pd.concat([stats_time[i], df])

                for i, df in enumerate(scount):
                    stats_count[i] = pd.concat([stats_count[i], df])
                
        else:
            stats_time, stats_count = validate(
                ground_truth = bnt_gt,
                to_validate = bnt_tv,
                binary = args.binary,
                positive_labels = positive_labels,
                late_start = args.late_start,
                early_stop = args.early_stop,
                overlapping_threshold_s = args.overlapping_threshold_s,
                skip_missing_gt=args.skip_missing_gt,
                gt_label_settings_path = gt_label_settings_path
            )

        def save_stats(stats: tuple[pd.DataFrame, pd.DataFrame], suffix: str):
            fname = lambda f: os.path.join(args.output_dir, f"{f}_{suffix}.csv")
            df_matrix, df_metrics  = stats
            df_matrix.to_csv(fname("confusion_matrix"))
            df_metrics.to_csv(fname("validation_metrics"))
        
        save_stats(stats_time, "time")
        save_stats(stats_count, "count")
    elif args.action=="calls":
        os.makedirs(args.output_dir, exist_ok=True)
        logger = Logger(logfile_path=os.path.join(args.output_dir, "log.txt"))
        annotations = Annotations(args.tables_dir, "bnrv")
        annotations.load()
        annotations.load_audio_paths(args.audio_files_dir)
        multi_processes(annotations, args.custom_classifier, args.output_dir, args.n_processes, logger)

        


    

        


