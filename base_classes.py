from pydub import AudioSegment
from pydub.utils import mediainfo
from dataclasses import dataclass
from math import ceil
from typing import Generator
import random
import numpy as np
import subprocess

import os
import csv


class TimeUnit(float):
    def __init__(self, s: float = 0, ms: float = None):
        if ms is not None:
            s = s / 1000
        self = float(s)

    def __add__(self, other):
        return TimeUnit(super().__add__(other))
    
    def __mul__(self, other):
        return TimeUnit(super().__mul__(other))
    
    def __sub__(self, other):
        return TimeUnit(super().__sub__(other))

    def __truediv__(self, other):
        return TimeUnit(super().__truediv__(other))

    
    @property
    def s(self):
        return self
    
    @property
    def ms(self):
        return int(self * 1000)

BIRDNET_AUDIO_DURATION = TimeUnit(3)

class Detection:
    tstart: TimeUnit
    tend: TimeUnit
    label: str

    def __init__(self, tstart_s: float, tend_s: float, label):
        self.tstart = TimeUnit(float(tstart_s))
        self.tend = TimeUnit(float(tend_s))
        self.label = label    
        

class DurDetection(Detection):
    def __init__(self, tstart, dur, label, audio_file):
        super().__init__(tstart, tstart+dur, label, audio_file)    

class AudioFile:
    audio_segment: AudioSegment = None
    exported_mask: np.ndarray
    
    def __init__(self, path: str):
        self.path = path
        self.basename = os.path.basename(path).split(".")[0]

    def load(self):
        if self.audio_segment is None:
            self.audio_segment: AudioSegment = AudioSegment.from_file(self.path)
            self.exported_mask = np.zeros(ceil(self.audio_segment.duration_seconds), dtype=np.bool_)
        
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
        out_path = os.path.join(dir_path, out_name)
        # self.audio_segment[int(tstart.ms):int(tend.ms)].export(out_path, tags=tags)
        subprocess.Popen(
            [
                "ffmpeg",
                "-i",
                self.path,
                "-ss",
                str(tstart.s),
                "-to",
                str(tend.s),
                out_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL 
        )
        if mark_exported:
            self.exported_mask[int(tstart.s):int(ceil(tend.s))] = True


    def export_for_birdnet(self, 
            detection: Detection, 
            base_path: str, 
            audio_format: str = "flac", 
            overlap_s: float = 0, 
            overlap_perc: float=0, 
            pad_to_birnet_duration: bool = True, 
            **kwargs
        ):

        dir_path: str = os.path.join(base_path, detection.label)
        os.makedirs(dir_path, exist_ok=True)

        tstart, tend = detection.tstart, detection.tend

        if overlap_perc > 0:
            overlap_s: float = BIRDNET_AUDIO_DURATION.s * overlap_perc
        
        overlap = TimeUnit(overlap_s)
        overlap = max(TimeUnit(0),min(overlap, BIRDNET_AUDIO_DURATION - TimeUnit(ms=1)))

        dur = tend - tstart

        if  dur <= BIRDNET_AUDIO_DURATION:
            if pad_to_birnet_duration:
                pad = (BIRDNET_AUDIO_DURATION - dur) / 2
                tstart -= pad
                tend += pad
            self.export_segment(tstart, tend, dir_path, audio_format, **kwargs)
            return
        
        tend_orig = tend
        start = True
        while start or tend < tend_orig:
            tend = tstart + BIRDNET_AUDIO_DURATION
            self.export_segment(tstart, min(tend_orig, tend), dir_path, audio_format, **kwargs)
            tstart = tend - overlap
            start = False

    def export_noise_birdnet(self, base_path: str, audio_format: str = "flac", export_prob: float = None, export_perc: float = .1, **kwargs):
        not_exp = ~self.exported_mask
        diff = np.diff(not_exp.astype(np.int8), prepend=0, append=0)
        noise_ratio = np.sum(not_exp) / len(not_exp)
        if noise_ratio == 0:
            return
        if export_prob is None:
            export_perc = max(min(1, export_perc), 0)
            print(noise_ratio)
            export_prob = min(1, export_perc / noise_ratio)
        tstarts = np.flatnonzero(diff == 1)
        tends = np.flatnonzero(diff == -1)
        
        random.seed(self.basename)

        dir_path: str = os.path.join(base_path, "Noise")
        os.makedirs(dir_path, exist_ok=True)
        

        for ts, te in zip(tstarts, tends):
            n_splits  = int((te-ts) / BIRDNET_AUDIO_DURATION.s)
            indexes = np.arange(n_splits) 
            k = int(round(n_splits * export_prob))
            indexes_to_export = random.choices(indexes, k = k)
            for i in indexes_to_export:
                tstart = TimeUnit(s=ts) + i * BIRDNET_AUDIO_DURATION
                tend = tstart + BIRDNET_AUDIO_DURATION
                self.export_segment(tstart, tend, dir_path, audio_format, mark=False, **kwargs)
        

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

    def get_detections(self, table_path: str, *args, **kwargs) -> Generator[Detection, None, None]:
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
        self.all_audio_files.setdefault(audio_path, audio_file)
        while True:
            yield audio_file

    def is_table(self, table_path: str):
        fname = os.path.basename(table_path)
        return fname.endswith(f".{self.file_ext}")