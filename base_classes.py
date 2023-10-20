from pydub import AudioSegment
from pydub.utils import mediainfo
from dataclasses import dataclass
from math import ceil
from typing import Generator
import numpy as np

import os
import csv

BIRDNET_AUDIO_DURATION_MS = 3000

class TimeUnit:
    s: float
    ms: float

    def __init__(self, s):
        self.s = s
        self.ms = s * 1000

class Detection:
    tstart: float
    tend: float
    label: str

    def __init__(self, tstart, tend, label):
        self.tstart = TimeUnit(float(tstart))
        self.tend = TimeUnit(float(tend))
        self.label = label    
        

class DurDetection(Detection):
    def __init__(self, tstart, dur, label, audio_file):
        super().__init__(tstart, tstart+dur, label, audio_file)    

class AudioFile:
    audio_segment: AudioSegment
    exported_intervals: list[list]
    exported_mask: np.ndarray
    
    def __init__(self, path):
        self.path = path
        self.basename = os.path.basename(path).split(".")[0]
        self.exported_intervals = 0

    def load(self):
        if self.audio_segment is None:
            self.audio_segment: AudioSegment = AudioSegment.from_file(self.path)
            self.exported_mask = np.array(self.audio_segment.duration_seconds)
        
    def export_segment(self, tstart: TimeUnit, tend: TimeUnit, dir_path: str, audio_format: str, mark_exported: bool = True, export_metadata: bool = True):
        self.load()
        tags = None
        if export_metadata:
            tags = mediainfo(self.path)['TAG']
        out_name = f"{self.basename}_{tstart.s:06.0f}_{tend.s:06.0f}.{audio_format}"
        out_path = os.path.join(dir_path, out_name)
        self.audio_segment[int(tstart.ms):int(tend.ms)].export(out_path, tags=tags)
        if mark_exported:
            self.exported_mask[int(tstart.s):int(ceil(tend.s))] = True


    def export_for_birdnet(self, detection: Detection, base_path: str, audio_format: str = "flac", overlap_s: float = 0, overlap_perc: float=0, **kwargs):
        dir_path: str = os.path.join(base_path, detection.label)
        os.makedirs(dir_path, makedirs=True)

        tstart, tend = detection.tstart_ms, detection.tend_ms

        if overlap_perc > 0:
            overlap_ms = int(BIRDNET_AUDIO_DURATION_MS * overlap_perc)
        overlap_ms = max(0, min(overlap_ms, BIRDNET_AUDIO_DURATION_MS - 1))

        if tend - tstart <= BIRDNET_AUDIO_DURATION_MS:
            self.export_segment(tstart, tend, dir_path, audio_format)
            return
        
        tend_orig = tend
        start = True
        while start or tend < tend_orig:
            tend = tstart + BIRDNET_AUDIO_DURATION_MS
            self.export_segment(tstart, min(tend_orig, tend), dir_path, audio_format)
            tstart = tend - overlap_ms
            start = False

        self.export_segment(tstart, tend, dir_path, **kwargs)
    


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
    file_ext: str = "csv"

    def __post_init__(self):
        self.columns = [self.tstart, self.tend, self.label]
        self.all_columns = self.columns
        self.all_audio_files = {}

    def set_coli(self, *args, **kwargs):
        if self.header:
            for col in self.columns:
                col.set_coli(*args, **kwargs)

    def get_detection(self, row: list, *args, **kwargs) -> Detection:
        return self.detection_type(
            *[col.get_val(row) for col in self.all_columns]
        )

    def get_detections(self, table_path: str, *args, **kwargs) -> Generator[Detection]:
        with open(table_path) as fp:
            csvr = csv.reader(fp, delimiter=self.delimiter)
            if self.header:
                theader = next(csvr)
                self.set_coli(theader)
            for row in csvr:
                yield self.get_detection(row, table_path, *args, **kwargs)
    
    def get_audio_files_paths(self, table_path: str, audio_file_dir_path: str, audio_file_ext: str) -> Generator[AudioFile]:
        table_basename: str = os.path.basename(table_path)
        base_path: str = os.path.join(audio_file_dir_path, table_basename.split(".")[0])
        audio_path: str = ".".join([base_path, audio_file_ext])
        audio_file = AudioFile(audio_path)
        while True:
            yield audio_file

    def is_table(self, table_path: str):
        fname = os.path.basename(table_path)
        return fname.endswith(f".{self.file_ext}")