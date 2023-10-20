from parsers import available_parsers
from base_classes import TableParser, AudioFile
import numpy as np
import fnmatch
import os


def get_parser(table_format: str, **parser_kwargs):
    table_format_name = table_format.lower()
    table_parser: TableParser = None
    for tp in available_parsers:
        tp_init: TableParser = tp(**parser_kwargs)
        if table_format_name in tp_init.names:
            table_parser = tp_init
    return table_parser

class BirdNetExtractor:
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
                if os.path.isfile(fpath) and fnmatch.fnmatch(fpath, self.parser.table_fnmatch):
                    self.tables_paths.append(fpath)
        

    def extract_detections_from_table(self, table_path: str, audio_files_dir: str, audio_file_ext: str, export_dir: str, **kwargs):
        detections = self.parser.get_detections(table_path, **kwargs)
        audiofiles = self.parser.get_audio_files(table_path, audio_files_dir, audio_file_ext)

        for det, af in zip(detections, audiofiles):
            af.export_for_birdnet(det, export_dir, **kwargs)


    def extract_all_detections(self, audio_files_dir: str, audio_file_ext: str, export_dir: str,  **kwargs):
        for tp in self.tables_paths:
            self.extract_detections_from_table(tp, audio_files_dir, audio_file_ext, export_dir, **kwargs)


    def extract_noise_all_files(self, export_dir: str, **kwargs):
        for af in self.parser.all_audio_files.values():
            af.export_noise_birdnet(**kwargs)


    def extract_for_training(self, audio_files_dir: str, audio_file_ext: str, export_dir: str,  **kwargs):
        self.extract_all_detections(audio_files_dir, audio_file_ext, export_dir, **kwargs)
        self.extract_noise_all_files(export_dir, **kwargs)





if __name__ == "__main__":
    extr = BirdNetExtractor(
        "raven",
        "C:\\Users\\plaf\\Documents\\raven",
        False
    )

    extr.extract_for_training(
        audio_files_dir="C:\\Users\\plaf\\Music",
        audio_file_ext="wav",
        export_dir="C:\\Users\\plaf\\Documents\\raven\\out"
    )
    

