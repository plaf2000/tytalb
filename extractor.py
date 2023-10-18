import csv
from pydub import AudioSegment
from dataclasses import dataclass
import os

# class NotAnAudioFile(Exception):
#     def __init__(self):
#         super().__init__("The file you provided is not a readable audio file.")

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



class Detection:
    tstart: float
    tend: float
    label: str
    audio_file: AudioFile

    def __init__(self, tstart, tend, label, audio_file):
        self.tstart = float(tstart)
        self.tend = float(tend)
        self.label = label
        self.audio_file = audio_file

    @property
    def tstart_ms(self):
        return self.tstart * 1000

    @property
    def tend_(self):
        return self.tend * 1000
    

class Column:
    colname: str = None
    colindex: int = None
    head: bool

    def __init__(self, colname, colindex, head):
        self.head = head
        if head:
            self.colname = colname
        else:
            self.colindex = colindex


    def set_coli(self, head_row: list):
        if self.head:
            self.colindex = head_row.index(self.colname)

    def get_val(self, row: list):
        return row[self.colindex]

    

@dataclass
class TableFormat:
    names: list[str]
    delimiter: str
    tstart: Column
    tend: Column
    label: Column
    file: Column = None
    head: bool

    def __post_init__(self):
        self.columns = [self.tstart, self.tend, self.label]

    def set_coli(self, *args, **kwargs):
        for col in self.columns:
            col.set_coli(*args, **kwargs)

    def read_row(self, row: list, audio_file: AudioFile) -> Detection:
        if self.file is not None:
            audio_file = AudioFile(self.file.get_val(row))
        return Detection(
            *[col.get_val(row) for col in self.columns],
            audio_file=audio_file,
        )
    
    def get_audio_file(self, table_path: str):
        if self.file is None:
            return table_path.split(".")[:-1]           


    def get_detections(self, table_path):
        detections = []
        with open(table_path) as fp:
            csvr = csv.reader(fp, delimiter=self.delimiter)
            if self.head:
                thead = next(csvr)
                self.set_coli(thead)
            for row in csvr:
                detections.append(self.read_row(row))
            

class KaleidoscopeParser(TableFormat):
    names = ["Kaleidoscope"]
    delimiter = ","
    tstart = "OFFSET"
    tend_colname = "DURATION"

    def read_row(self, row: list, _) -> Detection:
        tstart = self.
                
        
        

class DetectionsParser:
    raven_names = ["raven"]
    audacity_names = ["raven"]
    kaleidoscope_names = ["kaleidoscope"]
    sonic_names = ["sonic-visualizer", "sonic", "sv"]

    table_formats = [
        TableFormat(["raven"])
    ]

    def __init__(self, table_format: str):
        self.table_format_name = table_format.lower()
        self.table_format = None
        for tf in self.table_formats:
            if self.table_format_name == tf:
                self.table_format = tf
        if self.table_format is None:
            raise Exception("Table format not found")
        
    def get_detections(self, table_path):
        return self.table_format.get_detections(table_path)

        

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