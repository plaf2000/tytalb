from parsers import available_parsers, ap_names
from generic_parser import TableParser
from audio_file import AudioFile
from loggers import ProcLogger, Logger, ProgressBar 
from variables import BIRDNET_AUDIO_DURATION
from segment import Segment
from argparse import ArgumentParser
from datetime import datetime
import json
import numpy as np
import fnmatch
import time
import re
import os

def get_parser(table_format: str, **parser_kwargs):
    table_format_name = table_format.lower()
    table_parser: TableParser = None
    for tp in available_parsers:
        tp_init: TableParser = tp(**parser_kwargs)
        if table_format_name in tp_init.names:
            table_parser = tp_init
    return table_parser

class LabelMapper:
    def __init__(self, label_settings_path: str, *args, **kwargs):
        try:
            with open(label_settings_path) as fp:
                self.json_obj = json.load(fp)
        except:
            self.json_obj = {}
        self.map_dict: dict = {} if not "map" in self.json_obj else self.json_obj["map"]
        self.whitelist = None if not "whitelist" in self.json_obj else self.json_obj["whitelist"]
        self.blacklist = None if not "blacklist" in self.json_obj else self.json_obj["blacklist"]
    
    def black_listed(self, label: str) -> bool:
        if self.whitelist:
            return label not in self.whitelist
        return label in self.blacklist
    
    def map(self, label: str) -> str:
        for k, v in self.map_dict.items():
            if re.match(k, label):
                label = v
                break
        if self.black_listed(label):
            label = "Noise"
        return label
                


class BirdNetTrainer:
    tables_paths: list = []

    def __init__(
            self,
            table_format: str,
            tables_dir: str,
            recursive_subfolders = True,
            **parser_kwargs
        ):

        self.parser = get_parser(table_format, **parser_kwargs)
        self.tables_dir = tables_dir
        self.recursive_subfolders = recursive_subfolders
        if recursive_subfolders:
            for dirpath, dirs, files in os.walk(self.tables_dir): 
                for filename in fnmatch.filter(files, self.parser.table_fnmatch):
                    fpath = os.path.join(dirpath, filename)
                    self.tables_paths.append(fpath)
        else:
            for f in os.listdir(self.tables_dir):
                fpath = os.path.join(self.tables_dir, f)
                if os.path.isfile(fpath) and self.parser.is_table(fpath):
                    self.tables_paths.append(fpath)

    @property
    def n_tables(self):
        return len(self.tables_paths)

    def extract_for_training(self, audio_files_dir: str, audio_file_ext: str, export_dir: str, logger: Logger, **kwargs) -> dict[str, str | int | float]:
        logger.print("Input annotations' folder:", self.tables_dir)
        logger.print("Input audio folder:", audio_files_dir)
        logger.print("Output audio folder:", export_dir)
        self.map_audiofile_segments: dict[AudioFile, list] = {}
        segments: list[Segment] = []
        audiofiles: list[AudioFile] = []
        prog_bar = ProgressBar("Reading tables", len(self.tables_paths))
        for table_path in self.tables_paths:
            segments += self.parser.get_segments(table_path, **kwargs)
            audiofiles += self.parser.get_audio_files(table_path, audio_files_dir, audio_file_ext)
            prog_bar.print(1)
        prog_bar.terminate()

        if "label_settings_path" in kwargs:
            prog_bar = ProgressBar("Changing labels", len(segments))
            label_mapper = LabelMapper(**kwargs)
            for seg in segments:
                seg.label = label_mapper.map(seg.label)
                prog_bar.print(1)
            prog_bar.terminate()

        prog_bar = ProgressBar("Mapping annotations to audio files", len(segments))
        for seg, af in zip(segments, audiofiles):
            af.set_date(**kwargs)
            self.map_audiofile_segments.setdefault(af, []).append(seg)
            prog_bar.print(1)
        prog_bar.terminate()


        prog_bar = ProgressBar("Exporting segments and noise", len(segments))
        proc_logger = ProcLogger(**kwargs)
        logger.print("Found", len(self.map_audiofile_segments), "audio files.")
        for af, segs in self.map_audiofile_segments.items():
            af.export_all_birdnet(export_dir, segs, proc_logger=proc_logger, logger=logger, progress_bar=prog_bar, **kwargs)
        prog_bar.terminate()





if __name__ == "__main__":

    ts = time.time()
    arg_parser = ArgumentParser()
    arg_parser.description = f"Train a custom BirdNet classifier based on given annotations by first exporting "\
                             f"{BIRDNET_AUDIO_DURATION.s:.0f}s segments."
    
    subparsers = arg_parser.add_subparsers(dest="action")

    """
        Parse arguments to extract audio chunks
    """

    extract_parser = subparsers.add_parser("extract",
                                           help="Extracts audio chunks from long aduio files using FFmpeg based on the given parser annotation." \
                                                "the result consists of multiple audio file, each 3s long \"chunk\", each in the corresponding." \
                                                "labelled folder, which can be used to train the BirdNet custom classifier.")
    
    extract_parser.add_argument("-i", "--input-dir",
                                dest="tables_dir",
                                help="Path to the file or folder of the (manual) annotations.",
                                default=".")
    
    extract_parser.add_argument("-re", "--recursive",
                                type=bool,
                                dest="recursive",
                                help="Wether to look for tables inside the root directory recursively or not.",
                                default=True)
    
    extract_parser.add_argument("-f", "--annotation-format",
                                dest="table_format",
                                choices=ap_names,
                                required=True,
                                help="Annotation format to read the data inside the tables.")
    
    extract_parser.add_argument("-a", "--audio-root-dir",
                                dest="audio_files_dir",
                                help="Path to the root directory of the audio files. Default=current working dir.", default=".")
    
    extract_parser.add_argument("-ie", "--audio-input-ext",
                                dest="audio_input_ext",
                                required=True,
                                help="Key-sensitive extension of the input audio files.", default="wav")
    
    extract_parser.add_argument("-oe", "--audio-output-ext",
                                dest="audio_output_ext",
                                help="Key-sensitive extension of the output audio files.", default="flac")
    

    extract_parser.add_argument("-o", "--output-dir",
                                dest="export_dir",
                                help="Path to the output directory. If doesn't exist, it will be created.",
                                default=".")
    
    extract_parser.add_argument("-r", "--resample",
                                dest="resample",
                                help="Resample the chunk to the given value in Hz. (default = 48000)",
                                type=int,
                                default=48000)
    
    extract_parser.add_argument("-df", "--date-fromat",
                                dest="date_format",
                                help='Date format of the file. (default = "%%Y%%m%%d_%%H%%M%%S")',
                                type=str,
                                default="%Y%m%d_%H%M%S")
    
    args = arg_parser.parse_args()


    if args.action == "extract":
        export_dir = os.path.join(args.export_dir, datetime.utcnow().strftime("%Y%m%d_%H%M%SUTC"))
        os.mkdir(export_dir)
        
        # export_dir =args.export_dir
        

        logger = Logger(logfile_path=os.path.join(export_dir, "log.txt"))

        logger.print("Started processing...")
        ts = time.time()


        try:
            bnt = BirdNetTrainer(
                table_format=args.table_format,
                tables_dir=args.tables_dir,
                recursive_subfolders=args.recursive,
            )\
            .extract_for_training(
                audio_files_dir=args.audio_files_dir,
                audio_file_ext=args.audio_input_ext,
                export_dir=export_dir,
                audio_format=args.audio_output_ext,
                logger=logger,
                logfile_errors_path=os.path.join(export_dir, "error_log.txt"),
                logfile_success_path=os.path.join(export_dir, "success_log.txt"),
                label_settings_path = os.path.join(args.tables_dir, "labels.json"),
                resample=args.resample,
                date_format=args.date_format
            )
        
            logger.print(f"... end of processing (elapsed {time.time()-ts:.1f}s)")
        except Exception as e:
            print("An error occured and the operation was not completed!")
            print(f"Check {logger.logfile_path} for more information.")
            logger.print_exception(e)  

