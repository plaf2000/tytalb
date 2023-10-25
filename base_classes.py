from pydub import AudioSegment
from pydub.utils import mediainfo
from dataclasses import dataclass
from math import ceil
from typing import Generator, IO
import random
import numpy as np
import tempfile
import subprocess
from datetime import datetime, timedelta
import sys
import re
import fnmatch

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
    
    
    def time_str(self, write_all=False) -> str:
        s = round(self.s, ndigits=3)
        h = s // 3600
        s -= h * 3600
        m = s // 60
        s -= m*60
        if write_all or h>0:
            return f"{h:02.0f}:{m:02.0f}:{s:06.03f}"
        if m>0:
            return f"{m:02.0f}:{s:06.03f}"
        if s>0:
            return f"{s:06.03f}"

BIRDNET_AUDIO_DURATION = TimeUnit(3)

class Segment:
    tstart: TimeUnit
    tend: TimeUnit
    label: str |None

    def __init__(self, tstart_s: float, tend_s: float, label: str = None):
        self.tstart = TimeUnit(float(tstart_s))
        self.tend = TimeUnit(float(tend_s))
        self.label = label    
        
    @property
    def dur(self) -> TimeUnit:
        return self.tend - self.tstart
    
    def centered_pad(self, pad: TimeUnit):
        return Segment(self.tstart - pad, self.tend + pad, self.label)

    def centered_pad_to(self, duration: TimeUnit):
        return self.centered_pad((duration - self.dur)/2)

    def birdnet_pad(self):
        if self.dur > BIRDNET_AUDIO_DURATION:
            return self
        return self.centered_pad_to(BIRDNET_AUDIO_DURATION)

    def __str__(self):
        seg_name = f" \"{self.label}\"" if self.label is not None else ""
        return f"Segment{seg_name}: [{self.tstart.time_str(True)}, {self.tend.time_str(True)}]"
    
    

class DurSegment(Segment):
    def __init__(self, tstart, dur, label):
        super().__init__(tstart, tstart+dur, label)    


class ProgressBar():
# Thanks to https://stackoverflow.com/a/37630397
    
    def __init__(self, text, total, n_threads, bar_length=40):
        self.total = total
        self.bar_length = bar_length
        self.amount = 0
        self.increment=total/100//n_threads
        self.text = text
        self.print(increment=0)

    def is_to_update(self, i, verbose):
                i+=1
                return verbose and not i%self.increment
    
    def print(self, increment=None):
                if increment is None:
                    increment = self.increment
                self.amount+=increment
                if self.amount > self.total:
                    self.amount = self.total

                fraction = self.amount / self.total

                arrow = int(fraction * self.bar_length - 1) * '-' + '>'
                padding = int(self.bar_length - len(arrow)) * ' '

                ending = '\n' if self.amount == self.total else '\r'

                print(f'{self.text}: [{arrow}{padding}] {int(fraction*100)}%', end=ending)

    def terminate(self):
        self.print(self.total - self.amount)


class Logger:
    def __init__(self, log_type: str = "all", logfile_success: IO = None, logfile_errors: IO = None):
        self.log_errors = log_type == "errors" or log_type == "all"
        self.log_success = log_type == "all"
        self.logfile_success = logfile_success
        self.logfile_errors = logfile_errors
        self.errors = 0
        self.successes = 0

    def print_error(self, message):
        if self.log_errors:
            print(message, file=self.logfile_errors)
    
    def print_success(self, message):
        if self.log_success:
            print(message, file=self.logfile_success)

    def log_process(self, process: subprocess.CompletedProcess, success_message: str, error_message: str):
        try:
            process.check_returncode()
            self.print_success(success_message)
            self.successes += 1
        except subprocess.CalledProcessError as e:
            self.print_error(error_message)
            self.print_error(f"\t{e}")
            error_lines: list[str] = process.stderr.decode('utf_8').splitlines()
            for err in error_lines:
                self.print_error(f"\t{err}")
            self.errors += 1
        





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

    def segment_path(self, base_path: str, segment: Segment, audio_format = "flac") -> str:
        out_path = os.path.join(base_path, segment.label)
        os.makedirs(out_path, exist_ok=True)
    
        if self.date_time is not None:
            date = self.date_time + timedelta(seconds = segment.tstart.s)
            name = f"{self.prefix}{date.strftime(self.date_format)}_{segment.dur.s:05.0f}{self.suffix}.{audio_format}"
        else:
            name = f"{self.basename}_{segment.tstart.s:06.0f}_{segment.tend.s:06.0f}.{audio_format}"
        return os.path.join(out_path, name)

    @staticmethod
    def export_segment_ffmpeg(
            path: str,
            out_path: str,
            ss: TimeUnit = None,
            to: TimeUnit = None,
            ss_s: float = None,
            to_s: float = None,
            overwrite = True, **kwargs
        ) -> subprocess.CompletedProcess:


        args = ["ffmpeg", "-i", path, "-loglevel", "error"]
        if ss_s is not None:
            ss = TimeUnit(ss_s)
        if ss is not None:
            args += ["-ss", str(ss.s)]
        if to_s is not None:
            to = TimeUnit(to_s)
        if to is not None:
            args += ["-to", str(to.s)]

        args.append(out_path)

        if overwrite:
            args.append("-y")
        else: 
            args.append("-n")

        return subprocess.run(args, capture_output=True)

    @staticmethod
    def export_segments_ffmpeg(
            path: str,
            out_path: str,
            times: float | list[float],
            ss: TimeUnit=None,
            to: TimeUnit=None,
            ss_s: float=None,
            to_s: float=None,
            segment_list: str = None,
            segment_list_prefix: str = None,
            overwrite: bool =True,
            **kwargs
        ) -> subprocess.CompletedProcess:


        if ss_s is not None:
            ss = TimeUnit(ss_s)
        if to_s is not None:
            to = TimeUnit(to_s)
        
        args = ["ffmpeg", "-i", path, "-loglevel", "error"]

        if ss is not None:
            args+=["-ss", str(ss.s)]
        if to is not None:
            args+=["-to", str(to.s)]

        args += ["-f", "segment"]

        if isinstance(times, list) or isinstance(times, np.ndarray):
            args += ["-segment_times", ",".join([str(t) for t in times])]
        else:
            args += ["-segment_time", str(times)]
        
        if segment_list is not None:
            args += ["-segment_list", segment_list]
        if segment_list_prefix is not None:
            args += ["-segment_list_entry_prefix", segment_list_prefix]

        args.append(out_path)

        if overwrite:
            args.append("-y")
        else: 
            args.append("-n")

        return subprocess.run(args, capture_output=True)



    
    def export_all_birdnet(
            self,
            base_path: str,
            segments: list[Segment],
            audio_format: str = "flac",
            overlap_s: float = 0, 
            length_threshold_s=100 * BIRDNET_AUDIO_DURATION.s,
            logger: Logger = Logger(),
            **kwargs):
        """
        Export every segment to the corresponding folder in the `base_path` directory, each with the length defined 
        by the variable `BIRDNET_AUDIO_DURATION`.

        Parameters:
            - `base_path` (`str`): The base path where the exported audio clips will be saved.
            - `segments` (`list[Segment]`): A list of sound segments to export audio clips for.
            - `audio_format` (`str`, optional): The desired audio format for exported clips, as extension (default is "flac").
            - `overlap_s` (`float`, optional): The amount of overlap between segments in seconds for segments longer than `BIRDNET_AUDIO_DURATION` (default is 0).
            - `length_threshold_s` (`int`, optional): Length threshold in seconds above which the algorithm will start splitting 
              the long segments using the faster ffmpeg segment command, without overlap (default is 300).
            - `log_type` (`str`): which informations to log (default is `"all"`):
                - `"all"`: log all informations
                - `"errors"`: log errors only
                - `"none"`: do not log.
            - `logfile_success` (`IO`): used for writing successful operations (default is `None`, print to `sys.stdout`).
            - `logfile_errors` (`IO`): used for writing errors (default is `None`, print to `sys.stdout`).
            - `**kwargs`: Additional keyword arguments for customization.
        Example Usage:
        ```
        birdnet_instance.export_all_birdnet("./your/output/directory", segments, audio_format="wav", overlap_s=1)
        ```    

        """
        length_threshold = TimeUnit(length_threshold_s)
        segments = sorted([d.birdnet_pad() for d in segments], key=lambda seg: seg.tstart)
        overlap = TimeUnit(overlap_s)
        max_tend = 0
        for seg in segments:
            as_int = int(ceil(seg.tend.s))
            if  as_int > max_tend:
                max_tend = as_int

        # Boolean mask that for each second of the original file,
        # stores whether at least one segment overlaps or not.
        is_segment = np.zeros(max_tend + 1, np.bool_)

        for seg in segments:
            start = int(seg.tstart.s)
            end = int(ceil(seg.tend.s))
            is_segment[start:end] = True
        is_noise = ~is_segment
        diff = np.diff(is_noise.astype(np.int8), prepend=1, append=1)

        # Timestamps where the insterval changes from being 
        # segments-only to noise-only (and vice versa).
        tstamps = np.flatnonzero(np.abs(diff) == 1)

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmppath = lambda fname: os.path.join(tmpdirname, fname)
            listpath = tmppath(f"{self.basename}.csv")
            segment_path = tmppath(f"{self.basename}%04d.{audio_format}")

            # Start by efficiently splitting the large data file into smaller chunks of contiguous noise/segments.

            self.export_segments_ffmpeg(
                path=self.path,
                out_path=segment_path,
                times = tstamps,
                segment_list=listpath,
                segment_list_prefix=tmppath(""),
                **kwargs
            )

            seg = segments.pop(0)
            
            with open(listpath) as fp:
                csv_reader = csv.reader(fp)

                for row in csv_reader:
                    # For each chunk obtained

                    tstart_f = TimeUnit(row[1])
                    tend_f = TimeUnit(row[2])

                    if (tend_f-tstart_f).s < 3:
                        continue

                    fpath = row[0]
                    noise = True
                    while seg and seg.tend < tend_f:
                        # Until there are segments in the list within the chunk range

                        ss = seg.tstart - tstart_f
                        to = seg.tend - tstart_f
                        
                        if seg.dur.s <= BIRDNET_AUDIO_DURATION:
                            # If the duration of the segment is smaller or (most likely, since we added padding) equal
                            # to the audio duration allowed by BirdNet, export the entire segment without any further processing.
                            out_path = self.segment_path(base_path, seg, audio_format)
                            proc = self.export_segment_ffmpeg(path=fpath, ss=ss, to=to, out_path=out_path, **kwargs)
                            logger.log_process(
                                proc,
                                f"{seg} from file {self.path} exported to: {out_path}",
                                f"Error while exporting {seg}:"
                            )
                                             
                        elif seg.dur > length_threshold or overlap == 0:
                            # If the segment is very long (above length_threshold) or overlap is 0, use the faster segment ffmpeg command
                            # In this case we won't have any overlap (this command doesn't allow to do so).

                            splits = self.segment_path(base_path, seg, audio_format).split(".")
                            base_out_path = ".".join(splits[:-1])

                            out_path = f"{base_out_path}_%04d.{splits[-1]}"

                           
                            proc = self.export_segments_ffmpeg(path=fpath, ss=ss, to=to, times=BIRDNET_AUDIO_DURATION.s, out_path=out_path, **kwargs)
                            logger.log_process(
                                proc,
                                f"{BIRDNET_AUDIO_DURATION.s}s-segments test from {seg} in file {self.path} exported to: {out_path}",
                                f"Error while exporting {seg}:"
                            )


                            # Rename the files with more readable names (start and end time/date)
                            out_path = lambda i: f"{base_out_path}_{i:04d}.{splits[-1]}"
                            def new_out_path(sub_seg: Segment):
                                return self.segment_path(base_path, sub_seg, audio_format)
                            
                            i = 0
                            while os.path.isfile(out_path(i)):
                                tstart_seg = i * BIRDNET_AUDIO_DURATION + seg.tstart
                                tend_seg = tstart_seg + BIRDNET_AUDIO_DURATION
                                if  tend_seg > seg.tend:
                                    # Remove segment at the end that is shorter than birdnet duration
                                    os.remove(out_path(i))
                                    logger.print_success(f"Removed {out_path(i)}")
                                    i-=1
                                    break
                                # Rename files
                                noutp = new_out_path(Segment(tstart_seg, tend_seg, seg.label))
                                os.replace(out_path(i), noutp)
                                i+=1
                            logger.print_success(f"Renamed {i} segments.")

                            # Export the very last segment
                            to = seg.tend - tstart_f
                            ss = to - BIRDNET_AUDIO_DURATION

                            
                            sub_seg = Segment(ss + tstart_f, to + tstart_f, seg.label)
                            out_path = new_out_path(sub_seg)
                            proc = self.export_segment_ffmpeg(path=fpath, ss=ss, to=to, out_path=out_path, **kwargs)
                            logger.log_process(
                                proc,
                                f"{sub_seg} from {seg} in file {self.path} exported to: {out_path}",
                                f"Error while exporting {seg}:"
                            )
                            
                            

                        else:
                            start = True
                            last = False
                            while start or ss < to:
                                to_ = ss + BIRDNET_AUDIO_DURATION
                                if to_ > to:
                                    # If it is after the end of the segment, move the start back
                                    # so that the segment has the birdnet duration and terminates 
                                    # at the end of the segment. Set the last flag to true, since
                                    # moving forward doesn't make any sense (the entire segment is covered).

                                    ss -= to_ - to
                                    to_ = to
                                    last = True
                                start = False
                                sub_seg = Segment(ss + tstart_f, to_ + tstart_f, seg.label)
                                
                                out_path = self.segment_path(base_path, sub_seg, audio_format)
                                proc = self.export_segment_ffmpeg(path=fpath, ss=ss, to=to, out_path=out_path, **kwargs)
                                logger.log_process(
                                    proc,
                                    f"{sub_seg} from {seg} in file {self.path} exported to: {out_path}",
                                    f"Error while exporting {sub_seg} from {seg}:"
                                )

                                ss = to_ - overlap
                                if last:
                                    break
                        
                        # If we had any segment in this file chunk, 
                        # set the noise flag to False, so that this part of the 
                        # file is not identified as noise
                        noise = False

                        if not segments:
                            seg = None
                            break
                            
                        seg = segments.pop(0)                        
                        
                    if noise:
                        # If there are no segments in the file chunk,
                        # it can be splitted into small segments and exported
                        # as noise for training.

                        basename_noise_split = os.path.basename(fpath).split(".")
                        basepath_noise = os.path.join(base_path, "Noise")
                        os.makedirs(basepath_noise, exist_ok=True)
                        basepath_out = os.path.join(basepath_noise, ".".join(basename_noise_split[:-1]))

                        out_path = f"{basepath_out}%04d.{audio_format}"

                        
                        proc = self.export_segments_ffmpeg(path=fpath, times=BIRDNET_AUDIO_DURATION.s, out_path=out_path, **kwargs)
                        seg_noise = Segment(tstart_f, tend_f, "Noise")
                        logger.log_process(
                            proc,
                            f"{BIRDNET_AUDIO_DURATION.s}s-segment from {seg_noise} in file {self.path} exported to: {out_path}",
                            f"Error while exporting {seg_noise} into segments:"
                        )

                        # TODO: Change the noise filenames to more readable ones.                    

@dataclass
class Column:
    colname: str = None
    colindex: int = None

    def set_coli(self, header_row: list):
        """
        Finds the column in the header.
        """
        self.colindex = header_row.index(self.colname)

    def get_val(self, row: list):
        """
        Given the row's cells as list, return the value of the corresponding column.
        """
        return row[self.colindex]

@dataclass
class TableParser:
    names: list[str]
    delimiter: str
    tstart: Column
    tend: Column
    label: Column
    segment_type: Segment
    header: bool = True
    table_fnmatch: str = "*.csv"

    def __post_init__(self):
        # `self.columns` lists the columns used for retrieving the segment data (order is relevant!)
        self.columns: list[Column] = [self.tstart, self.tend, self.label]
        # `self.all_columns` lists all the columns, it is used to set the indices from the header
        self.all_columns: list[Column] = self.columns

        # Dictionary mapping from paths to `AudioFile` objects contained in the segment table
        # (usually, only one).
        self.all_audio_files: dict[str, AudioFile] = {}

    def set_coli(self, *args, **kwargs):
        """
        Set the column indices for columns that have headers.
        """
        if self.header:
            for col in self.all_columns:
                col.set_coli(*args, **kwargs)

    def get_segment(self, row: list) -> Segment:
        """
        Instantiate the `Segment` object by reading the row values.
        """
        return self.segment_type(
            *[col.get_val(row) for col in self.columns]
        )

    def get_segments(self, table_path: str, *args, **kwargs) -> Generator[Segment, None, None]:
        """
        Returns a generator that for each line of the table yields the segment.
        If the table has an header, it first sets the columns using the header.
        """
        with open(table_path) as fp:
            csvr = csv.reader(fp, delimiter=self.delimiter)
            if self.header:
                theader = next(csvr)
                self.set_coli(theader)
            for row in csvr:
                yield self.get_segment(row)
    
    def get_audio_files(self, table_path: str, audio_file_dir_path: str, audio_file_ext: str) -> Generator[AudioFile, None, None]:
        """
        Given the table path, the directory containing the audio file and the audio file
        extenstion, returns a generator that yields the audio file corresponding to each segment.

        By default, there is only one audio file per table, which is retrieved
        by looking in the audio directory for audio files that have the same 
        name as the table + the provided extension in the arguments.
        """
        table_basename = os.path.basename(table_path)
        base_path = os.path.join(audio_file_dir_path, table_basename.split(".")[0])
        audio_path = ".".join([base_path, audio_file_ext])
        audio_file = AudioFile(audio_path)
        self.all_audio_files.setdefault(audio_path, audio_file)

        with open(table_path) as fp:
            csvr = csv.reader(fp, delimiter=self.delimiter)
            for _ in csvr:
                yield audio_file

    def is_table(self, table_path: str) -> bool:
        """
        Returns if the provided path matches the tables' file name pattern.
        """
        basename = os.path.basename(table_path)
        return fnmatch.fnmatch(basename, self.table_fnmatch)
