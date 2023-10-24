from pydub import AudioSegment
from pydub.utils import mediainfo
from dataclasses import dataclass
from math import ceil
from typing import Generator
import random
import numpy as np
import tempfile
import subprocess
from datetime import datetime, timedelta
import re

import os
import csv

dateel_lengths = {
    "%Y": 4,
    "%y": 2,
    "%m": 2,
    "%d": 2,
    "%j": 3,
    "%H": 2,
    "%I": 2,
    "%p": 2,
    "%M": 2,
    "%S": 2,
    "%f": 6,
}

class TimeUnit(float):
    def __init__(self, s: float | str = 0, ms: float | str | None = None):
        if ms is not None:
            s = float(ms) / 1000
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
        
    @property
    def dur(self) -> TimeUnit:
        return self.tend - self.tstart
    
    def centered_pad(self, pad: TimeUnit):
        return Detection(self.tstart - pad, self.tend + pad, self.label)

    def centered_pad_to(self, duration: TimeUnit):
        return self.centered_pad((duration - self.dur)/2)

    def birdnet_pad(self):
        if self.dur > BIRDNET_AUDIO_DURATION:
            return self
        return self.centered_pad_to(BIRDNET_AUDIO_DURATION)
    
    

class DurDetection(Detection):
    def __init__(self, tstart, dur, label, audio_file):
        super().__init__(tstart, tstart+dur, label, audio_file)    

class AudioFile:
    audio_segment: AudioSegment = None
    exported_mask: np.ndarray
    
    def __init__(self, path: str):
        self.path = path
        self.basename = os.path.basename(path).split(".")[0]
        self.date_time = None
    
    def set_date(self, date_format:str = "%Y%m%d_%H%M%S"):
        self.date_format = date_format
        fre = date_format
        for k, v in dateel_lengths.items():
            fre = fre.replace(k,r"\d{" + str(v) + r"}")
        m = re.search(fre, self.basename)
        if m:
            self.prefix, self.suffix = re.split(fre, self.basename)
            self.date_time = datetime.strptime(m.group(0), date_format)

    def detection_path(self, base_path: str, detection: Detection, audio_format = "flac") -> str:
        out_path = os.path.join(base_path, detection.label)
        os.makedirs(out_path)
    
        if self.date_time is not None:
            date = self.date_time + timedelta(seconds = detection.dur.s)
            name = f"{self.prefix}{date.strftime(self.date_format)}{self.suffix}.{audio_format}"
        else:
            name = f"{self.basename}.{audio_format}"
        
        return os.path.join(out_path, name)


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
    
    def export_all_birdnet(self, base_path: str, detections: list[Detection], audio_format: str = "flac", overlap = 0):
        detections = sorted([d.birdnet_pad() for d in detections], key=lambda det: det.tstart)
        for det in detections:
            self.exported_mask[int(det.tstart.s):int(ceil(det.tend.s))] = True
        not_exp = ~self.exported_mask
        diff = np.diff(not_exp.astype(np.int8), prepend=0, append=0)

        tstamps = np.flatnonzero(np.abs(diff) == 1) # TODO: Check offset +- 1 
        tstamps_cmd = np.array2string(tstamps, prefix="", suffix="", separator=",")


        with tempfile.TemporaryDirectory() as tmpdirname:
            tmppath = lambda fname: os.path.join(tmpdirname, fname)
            listpath = tmppath(f"{self.basename}.csv")
            segment_path = tmppath(f"{self.basename}%04d.{audio_format}")

            subprocess.run(
                args=[
                    "ffmpeg",
                    "-i",
                    self.path,
                    "-f",
                    "segment",
                    "-segment_times",
                    tstamps_cmd,
                    "-segment_list",                    
                    listpath,
                    "-segment_list_entry_prefix",
                    tmpdirname,
                    segment_path
                ]
            )

            det = detections.pop(0)
            
            with open(listpath) as fp:
                csv_reader = csv.reader(fp)

                for row in csv_reader:
                    tstart_f = TimeUnit(s=row[1])
                    tend_f = TimeUnit(s=row[2])

                    if (tstart_f-tend_f).s < 3:
                        continue

                    fpath = row[0]

                    noise = True
                    while det.tstart < tend_f:
                        if det.dur.s <= BIRDNET_AUDIO_DURATION:
                            ss = det.tstart - tstart_f
                            to = det.tend - tstart_f
                            subprocess.run(
                                args = [
                                    "ffmpeg",
                                    "-i",
                                    fpath,
                                    "-ss",
                                    str(ss.s),
                                    "-to",
                                    str(to.s),
                                    self.detection_path(base_path, det, audio_format)
                                ]
                            )
                        
                        else:
                            ss = det.tstart - tstart_f
                            start = True
                            while start or ss < tend_f:
                                to = ss + BIRDNET_AUDIO_DURATION
                                ss = to - overlap
                                start = False
                                sub_det = Detection(ss + tstart_f, to + tstart_f, det.label)
                                subprocess.run(
                                    args = [
                                        "ffmpeg",
                                        "-i",
                                        fpath,
                                        "-ss",
                                        str(ss.s),
                                        "-to",
                                        str(to.s),
                                        self.detection_path(base_path, sub_det, audio_format)
                                    ]
                                )
                        noise = False
                        


                    if noise:
                        basename_noise_split = os.path.basename(fpath).split(".")
                        basepath_noise = os.path.join(base_path, "Noise")
                        os.makedirs(basepath_noise, exist_ok=True)
                        basepath_out = os.path.join(basepath_noise, ".".join(basename_noise_split[:-1]))

                        fout_name = f"{basepath_out}%04d.{audio_format}"

                        subprocess.run(
                            args=[
                                "ffmpeg",
                                "-i",
                                fpath,
                                "-f",
                                "segment",
                                "-segment_time",
                                str(BIRDNET_AUDIO_DURATION.s),
                                "-segment_list",                    
                                listpath,
                                "-segment_list_entry_prefix",
                                fout_name,                                
                            ]
                        )


                    

                    

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
