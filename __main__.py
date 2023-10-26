from parsers import available_parsers
from base_classes import TableParser, AudioFile, Logger, ProgressBar
from argparse import ArgumentParser
import numpy as np
import fnmatch
import time
import os


ap_names = []
for ap in available_parsers:
    ap_names += ap().names

def get_parser(table_format: str, **parser_kwargs):
    table_format_name = table_format.lower()
    table_parser: TableParser = None
    for tp in available_parsers:
        tp_init: TableParser = tp(**parser_kwargs)
        if table_format_name in tp_init.names:
            table_parser = tp_init
    return table_parser

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

    def extract_for_training(self, audio_files_dir: str, audio_file_ext: str, export_dir: str, **kwargs):
        self.map_audiofile_segments: dict[AudioFile, list] = {}
        segments = []
        audiofiles = []
        prog_bar = ProgressBar("Reading tables", len(self.tables_paths))
        for table_path in self.tables_paths:
            segments += self.parser.get_segments(table_path, **kwargs)
            audiofiles += self.parser.get_audio_files(table_path, audio_files_dir, audio_file_ext)
            prog_bar.print(1)
        prog_bar.terminate()

        prog_bar = ProgressBar("Mapping annotations to audio files", len(segments))
        for seg, af in zip(segments, audiofiles):
            self.map_audiofile_segments.setdefault(af, []).append(seg)
            prog_bar.print(1)
        prog_bar.terminate()


        prog_bar = ProgressBar("Exporting segments and noise", len(segments))
        with open("success.log", "w") as logfile_success:
            with open("test_err.log", "w") as logfile_errors:
                for af, segs in self.map_audiofile_segments.items():
                        logger = Logger(logfile_success=logfile_success, logfile_errors=logfile_errors)
                        af.export_all_birdnet(export_dir, segs, logger=logger, progress_bar=prog_bar, **kwargs)
        # prog_bar.terminate()





if __name__ == "__main__":

    ts = time.time()
    arg_parser = ArgumentParser()
    arg_parser.description = "Train a custom BirdNet classifier based on given annotations by first exporting 3s segments."
    
    subparsers = arg_parser.add_subparsers(dest="action")
    extract_parser = subparsers.add_parser("extract",
                                           help="Extracts audio chunks from long aduio files using FFmpeg based on the given parser annotation." \
                                                "the result consists of multiple audio file, each 3s long \"chunk\", each in the corresponding." \
                                                "labelled folder, which can be used to train the BirdNet custom classifier.")
    extract_parser.add_argument("-i", "--input-dir",
                                dest="tables_dir",
                                help="Path to the file or folder of the (manual) annotations.",
                                default=".")
    
    extract_parser.add_argument("-u", "--recursive",
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
    
    extract_parser.add_argument("-e", "--audio-input-ext",
                                dest="audio_input_ext",
                                required=True,
                                help="Key-sensitive extension of the input audio files.", default="wav")
    
    # extract_parser.add_argument("-e", "--audio-input-ext",
    #                             dest="audio_input_ext",
    #                             help="Key-sensitive extension of the input audio files.", default="flac")
    

    extract_parser.add_argument("-o", "--output-dir",
                                dest="output_dir",
                                help="Path to the output directory. If doesn't exist, it will be created.",
                                default=".")
    
    extract_parser.add_argument("-r", "--resample", help="Resample the chunk to the given value in Hz.", type=int,
                        default=48000)

    args = arg_parser.parse_args()


    if args.action == "extract":
        bnt = BirdNetTrainer(
            table_format=args.table_format,
            tables_dir=args.tables_dir,
            recursive_subfolders=args.recursive,
        )
        bnt.extract_for_training(
            audio_files_dir=args.audio_files_dir,
            audio_file_ext=args.audio_input_ext,
            export_dir=args.output_dir,
            # audio_format=args.out_audio_ext
        )


    # bnt = BirdNetTrainer(
    #     "raven",
    #     "C:\\Users\\plaf\\Documents\\raven",
    #     False
    # )

    # bnt.extract_for_training(
    #     audio_files_dir="C:\\Users\\plaf\\Music",
    #     audio_file_ext="wav",
    #     export_dir="C:\\Users\\plaf\\Documents\\raven\\out",
    #     audio_format="flac",
    #     resample=48000,
    # )

    # print(time.time() - ts)
    

