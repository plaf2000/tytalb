from parsers import available_parsers
from base_classes import TableParser
import os


def get_parser(table_format: str):
    table_format_name = table_format.lower()
    table_parser: TableParser = None
    for tf in available_parsers:
        if table_format_name == tf:
            table_parser = tf
    return table_parser

class BirdNetExtractor:
    tables_paths: list = []
    parser: TableParser

    def __init__(self, table_format: str, tables_dir: str,  audio_files_dir: str, recursive_subfolders = True):
        self.parser = get_parser(table_format)
        self.tables_dir = tables_dir
        self.audio_files_dir = audio_files_dir
        self.recursive_subfolders = recursive_subfolders
        paths = [os.path.join(self.tables_dir, f) for f in os.listdir(self.tables_dir)]
        while paths:
            p = paths.pop()
            if os.path.isfile(p):
                if self.parser.is_table(p):
                    self.tables_paths.append(p)
            elif recursive_subfolders and os.path.isdir(p):
                paths.append(p)

    def extract_from_table(self, table_path, audiofile_path):


    def extract_all(self):
        