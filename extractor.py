import csv
from pydub import AudioSegment
from dataclasses import dataclass
import os
from os import path

@dataclass
class Extractor():
    table_format: str
    tables_dir: str 
    recursive_subfolders: bool = True
    tables_paths: list = []

    def __post_init__(self):
        self.tables_dirs = []
        if self.recursive_subfolders:
            os.listdir(self.tables_dir)

    def extract_from_table(csv_path):
        fname = ""
        with open(csv_path) as fp:
            csvr = csv.reader(fp, delimiter="\t")
            thead = next(csvr)
            ts_i = thead.index("Begin Time (s)")
            te_i = thead.index("End Time (s)")

            for row in csvr:
                tstart = row[ts_i]
                tend = row[te_i]
                AudioSegment.from_file()