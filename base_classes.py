from pydub import AudioSegment
from dataclasses import dataclass
import os
import csv

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


    def export_for_birdnet(self, tstart, tend, dir_path, audio_format = "flac", overlap_ms = 0):
        if tstart - tend <= 3000:
            self.export_segment(tstart, tend, dir_path, audio_format)
            return
        
        tend_orig = tend
        while tend < tend_orig:
            tend = tstart 
            self.export_segment(tstart, tend, dir_path, audio_format)
            tstart = tend - overlap_ms
        

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

    def set_coli(self, header_row: list):
        self.colindex = header_row.index(self.colname)

    def get_val(self, row: list):
        return row[self.colindex]

@dataclass
class TableParser:
    names: list[str]
    delimiter: str
    tstart: Column
    tend: Column
    label: Column
    audio_file_path: list[Column] = None
    detection_type = Detection
    header: bool = True
    file_ext: str = "csv"

    def __post_init__(self):
        self.columns = [self.tstart, self.tend, self.label]

    def set_coli(self, *args, **kwargs):
        if self.header:
            for col in self.columns:
                col.set_coli(*args, **kwargs)

    def read_row(self, row: list, audio_file: AudioFile = None) -> Detection:
        if self.file is not None:
            full_path = os.path.join(*[p.get_val(row) for p in self.audio_file_path])
            audio_file = AudioFile(full_path)

        return self.detection_type(
            *[col.get_val(row) for col in self.columns],
            audio_file=audio_file,
        )
    
    def get_audio_file(self, table_path: str):
        if self.audio_file_path is None:
            return table_path.split(".")[:-1]           


    def get_detections(self, table_path) -> list[Detection]:
        detections = []
        with open(table_path) as fp:
            csvr = csv.reader(fp, delimiter=self.delimiter)
            if self.header:
                theader = next(csvr)
                self.set_coli(theader)
            for row in csvr:
                detections.append(self.read_row(row))
        return detections
    
    def is_table(self, table_path: str):
        fname = os.path.basename(table_path)
        return fname.endswith(f".{self.file_ext}")