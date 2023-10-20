from pydub import AudioSegment
from pydub.utils import mediainfo
from dataclasses import dataclass
from math import ceil
from typing import Generator
import numpy as np

import os
import csv


class TimeUnit:
    s: float
    ms: float

    def __init__(self, s: float = 0, ms: float = None):
        if ms is not None:
            s = s / 1000
        self.s = s
    def __float__(self):
        return self.s

    def __eq__(self, other):
        self.s == other.s

    def __gt__(self, other):
        self.s > other.s

    def __lt__(self, other):
        self.s < other.s

    def __ge__(self, other):
        self.s >= other.s

    def __le__(self, other):
        self.s <= other.s

    def __add__(self, other):
        return TimeUnit(self.s + other.s)
    
    def __sub__(self, other):
        return TimeUnit(self.s - other.s)

    def __mul__(self, other):
        return TimeUnit(self.s * other.s)
    
    def copy(self):
        return TimeUnit(self.s)
    
    @property
    def ms(self):
        return int(self.s * 1000)

BIRDNET_AUDIO_DURATION = TimeUnit(3)

class Detection:
    tstart: TimeUnit
    tend: TimeUnit
    label: str

    def __init__(self, tstart, tend, label):
        self.tstart = TimeUnit(float(tstart))
        self.tend = TimeUnit(float(tend))
        self.label = label    
        

class DurDetection(Detection):
    def __init__(self, tstart, dur, label, audio_file):
        super().__init__(tstart, tstart+dur, label, audio_file)    

class AudioFile:
    audio_segment: AudioSegment = None
    exported_intervals: list[list]
    exported_mask: np.ndarray
    
    def __init__(self, path):
        self.path = path
        self.basename = os.path.basename(path).split(".")[0]
        self.exported_intervals = 0

    def load(self):
        if self.audio_segment is None:
            self.audio_segment: AudioSegment = AudioSegment.from_file(self.path)
            self.exported_mask = np.zeros(int(ceil(self.audio_segment.duration_seconds)), dtype=np.bool_)
        
    def export_segment(
            self,
            tstart: TimeUnit,
            tend: TimeUnit,
            dir_path: str,
            audio_format: str,
            mark_exported: bool = True,
            export_metadata: bool = True,
            tstart_s = None,
            tend_s = None,
            tstart_ms = None,
            tend_ms = None,
            **kwargs
        ):

        if tstart_s is not None:
            tstart = TimeUnit(tstart_s)
        if tstart_ms is not None:
            tstart = TimeUnit(ms = tstart_ms)

        if tend_s is not None:
            tend = TimeUnit(tend_s)
        if tend_ms is not None:
            tend = TimeUnit(ms = tend_ms)

        self.load()
        tags = None
        if export_metadata:
            tags = mediainfo(self.path)['TAG']
        out_name = f"{self.basename}_{tstart.s:06.0f}_{tend.s:06.0f}.{audio_format}"
        out_path = os.path.join(dir_path,
        out_name)
        self.audio_segment[int(tstart.ms):int(tend.ms)].export(out_path, tags=tags)
        if mark_exported:
            self.exported_mask[int(tstart.s):int(ceil(tend.s))] = True


    def export_for_birdnet(self, detection: Detection, base_path: str, audio_format: str = "flac", overlap_s: float = 0, overlap_perc: float=0, **kwargs):
        dir_path: str = os.path.join(base_path, detection.label)
        os.makedirs(dir_path, exist_ok=True)

        tstart, tend = detection.tstart.copy(), detection.tend.copy()

        if overlap_perc > 0:
            overlap_s: float = BIRDNET_AUDIO_DURATION.s * overlap_perc
        
        overlap = TimeUnit(overlap_s)
        overlap = max(TimeUnit(0),min(overlap, BIRDNET_AUDIO_DURATION - TimeUnit(ms=1)))

        if tend - tstart <= BIRDNET_AUDIO_DURATION:
            self.export_segment(tstart, tend, dir_path, audio_format, **kwargs)
            return
        
        tend_orig = tend
        start = True
        while start or tend < tend_orig:
            tend = tstart + BIRDNET_AUDIO_DURATION
            self.export_segment(tstart, min(tend_orig, tend), dir_path, audio_format, **kwargs)
            tstart = tend - overlap
            start = False

    def export_noise_birdnet(self, base_path: str, audio_format: str = "flac", keep_perc = .1, **kwargs):
        diff = np.diff(self.exported_mask.astype(np.int8))
        tstarts = list(np.nonzero(diff == -1))
        tends = list(np.nonzero(diff == 1))

        





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
    detection_type: Detection
    header: bool = True
    table_fnmatch: str = "*.csv"

    def __post_init__(self):
        self.columns = [self.tstart, self.tend, self.label]
        self.all_columns = self.columns
        self.all_audio_files: dict[str, AudioFile] = {}

    def set_coli(self, *args, **kwargs):
        if self.header:
            for col in self.columns:
                col.set_coli(*args, **kwargs)

    def get_detection(self, row: list) -> Detection:
        return self.detection_type(
            *[col.get_val(row) for col in self.all_columns]
        )

    def get_detections(self, table_path: str) -> Generator[Detection, None, None]:
        with open(table_path) as fp:
            csvr = csv.reader(fp, delimiter=self.delimiter)
            if self.header:
                theader = next(csvr)
                self.set_coli(theader)
            for row in csvr:
                yield self.get_detection(row)
    
    def get_audio_files(self, table_path: str, audio_file_dir_path: str, audio_file_ext: str) -> Generator[AudioFile, None, None]:
        table_basename: str = os.path.basename(table_path)
        base_path: str = os.path.join(audio_file_dir_path, table_basename.split(".")[0])
        audio_path: str = ".".join([base_path, audio_file_ext])
        audio_file = AudioFile(audio_path)
        while True:
            yield audio_file

    def is_table(self, table_path: str):
        fname = os.path.basename(table_path)
        return fname.endswith(f".{self.file_ext}")