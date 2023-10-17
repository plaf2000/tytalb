import csv
from pydub import AudioSegment
from dataclasses import dataclass
import os

@dataclass
class Detection:
    tstart: float
    tend: float
    label: str

    @property
    def tstart_ms(self):
        return self.tstart * 1000

    @property
    def tend_(self):
        return self.tend * 1000


class AudioFile:
    audio_segment: AudioSegment = None
    
    def __init__(self, path):
        self.path = path

    def load(self):
        if self.audio_segment is None:
            self.audio_segment = AudioSegment.from_file(self.path)

    def export_segment(self, tstart, tend, out_path, *args):
        self.load()
        self.audio_segment[tstart:tend].export(out_path)
    
@dataclass
class TableFormat:
    names: list[str]
    delimiter: str
    tstart_field: str
    tend_field: str

class DetectionsParser:
    raven_names = ["raven"]
    audacity_names = ["raven"]
    kaleidoscope_names = ["kaleidoscope"]
    sonic_names = ["sonic-visualizer", "sonic", "sv"]

    def __init__(self, table_format: str):
        self.table_format = table_format.lower()
        if self.table_format in self.raven_names + self.audacity_names:
            self.delimiter = "\t"
        elif self.table_format in self.kaleidoscope_names + self.sonic_names:
            self.delimiter = ","

    
        

class BirdNetExtractor:
    tables_paths: list = []
    audiofiles_paths: list = []

    def __init__(self, tables_dir: str, recursive_subfolders = True):
        self.tables_dir = tables_dir
        self.recursive_subfolders = recursive_subfolders
        self.tables_paths = []
        if self.recursive_subfolders:
            os.listdir(self.tables_dir)

    def extract_from_table(self, table_path, audiofile_path):
        with open(table_path) as fp:
            csvr = csv.reader(fp, delimiter=self.delimiter)
            thead = next(csvr)
            ts_i = thead.index("Begin Time (s)")
            te_i = thead.index("End Time (s)")
            audio_file: AudioSegment = AudioSegment.from_file(audiofile_path)
                


            for row in csvr:
                tstart = int((float(row[ts_i])) * 1000)
                tend = int((float(row[te_i])) * 1000)

                if tend - tstart > 3000:
                    tend_orig = tend
                    while tend < tend_orig:
                        tend = tstart + 3000

                        tstart = tend

    def extract_all(self):
        for tpath, apath in zip(self.tables_paths, self.audiofiles_paths):
            self.extract_from_table(table_path=tpath, audiofile_path=apath)