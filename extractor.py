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
        self.basename = os.path.basename(path).split(".")[0]

    def load(self):
        if self.audio_segment is None:
            self.audio_segment = AudioSegment.from_file(self.path)

    def export_segment(self, tstart, tend, dir_path, audio_format):
        self.load()
        out_path = os.path.join(dir_path, f"{self.basename}_{tstart:09.0f}_{tend:09.0f}.{audio_format}")
        self.audio_segment[tstart:tend].export(out_path)


    def export_for_birdnet(self, tstart, tend, dir_path, audio_format = "flac", overlap_long = False):
        def full_name(ts, te):
            return
        if tstart - tend <= 3000:
            self.export_segment(tstart, tend, dir_path, audio_format)
            return
        
        tend_orig = tend
        while tend < tend_orig:
            tend = tstart 
            self.export_segment(tstart, tend, dir_path, audio_format)
            tstart = tend
            if overlap_long:
                tstart -= 500
        

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
    def tend_ms(self):
        return self.tend * 1000
    
    def export_for_birdnet(self, base_path, **kwargs):
        dir_path = os.path.join(base_path, self.label)
        if not os.path.isdir(dir_path):
            os.mkdir(base_path)
        self.audio_file.export_for_birdnet(self.tstart_ms, self.tend_ms, **kwargs)

    
class DurDetection(Detection):
    def __init__(self, tstart, dur, label, audio_file):
        super().__init__(tstart, tstart+dur, label, audio_file)    

@dataclass
class Column:
    colname: str = None
    colindex: int = None

    def set_coli(self, head_row: list):
        self.colindex = head_row.index(self.colname)

    def get_val(self, row: list):
        return row[self.colindex]

@dataclass
class TableParser:
    names: list[str]
    delimiter: str
    tstart: Column
    tend: Column
    label: Column
    file: list[Column] = None
    detection_type = Detection
    head: bool

    def __post_init__(self):
        self.columns = [self.tstart, self.tend, self.label]

    def set_coli(self, *args, **kwargs):
        if self.head:
            for col in self.columns:
                col.set_coli(*args, **kwargs)

    def read_row(self, row: list, audio_file: AudioFile = None) -> Detection:
        if self.file is not None:
            full_path = os.path.join(*[p.get_val(row) for p in self.file])
            audio_file = AudioFile(full_path)

        return self.detection_type(
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
            

class KaleidoscopeParser(TableParser):
    names = ["kaleidoscope", "ks"]
    delimiter = ","
    tstart = Column("OFFSET", 3)
    tend = Column("DURATION", 4)
    label = Column("scientific_name", 5)
    file =  [
        Column("INDIR", 0),
        Column("FOLDER", 1),
        Column("IN FILE", 2),
    ]
    head = True
    detection_type = DurDetection

class SonicParser(TableParser):
    names = ["sonic-visualizer", "sonic-visualiser", "sv"]
    delimiter = ","
    tstart = Column("START", 0)
    tend = Column("END", 1)
    label = Column("LABEL", 4)
    head = False

class AudacityParser(TableParser):
    names = ["sonic-visualizer", "sonic-visualiser", "sv"]
    delimiter = "\t"
    tstart = Column(None, 0)
    tend = Column(None, 1)
    label = Column(None, 2)
    head = False

class RavenParser(TableParser):
    names = ["raven", "rvn"]
    delimiter = "\t"
    tstart = Column("Begin Time (s)", 3)
    tend = Column("End Time (s)", 4)
    label = Column("Annotation", 10)
    head = True      

class DetectionsParser:
    table_parsers = [
        KaleidoscopeParser(),
        SonicParser(),
        AudacityParser(),
        RavenParser(),
    ]

    def __init__(self, table_format: str):
        self.table_format_name = table_format.lower()
        self.table_format: TableParser = None
        for tf in self.table_parsers:
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

    def extract_all(self):
        for tpath, apath in zip(self.tables_paths, self.audiofiles_paths):
            self.extract_from_table(table_path=tpath, audiofile_path=apath)