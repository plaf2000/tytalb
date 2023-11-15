import csv
from datetime import datetime, timedelta
import soundfile as sf
from math import ceil
import os
import tempfile
import random
import time
import numpy as np 
from segment import Segment
from units import TimeUnit
import subprocess
import re
from loggers import Logger, ProcLogger, ProgressBar

from variables import BIRDNET_AUDIO_DURATION, NOISE_LABEL, BIRDNET_SAMPLE_RATE


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

# TODO: Figure out the cause of the problem for the permission error
# ("The process cannot access the file because it is being used by another process")
def permission_safe(action, args):
    while True:
        try:
            action(*args)
            break
        except PermissionError:
            time.sleep(.1)


class AudioFile:
    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File {path} not found.")
        self.path = path
        self.splits = os.path.basename(path).split(".")
        self.basename = self.splits[0]
        self.ext = self.splits[-1].lower()
        self.duration = TimeUnit(sf.info(self.path).duration)
        self.date_time = None
    
    def set_date(self, date_format:str = "%Y%m%d_%H%M%S", **kwargs):
        """
        Sets the date time of the audio file according to the provided filename date format.
        """
        self.date_format = date_format
        fre = date_format
        for k, v in dateel_lengths.items():
            fre = fre.replace(k,r"\d{" + str(v) + r"}")
        m = re.search(fre, self.basename)
        if m:
            self.prefix, self.suffix = re.split(fre, self.basename)
            self.date_time = datetime.strptime(m.group(0), date_format)
            return
        self.prefix = f"{self.basename}_"

    def segment_path(self, base_path: str, segment: Segment, audio_format: str) -> str:
        """
        Returns the path for an audio segment, according to its label and (if provided) the date time of the
        original audio file.
        """
        out_path = os.path.join(base_path, segment.label)
        os.makedirs(out_path, exist_ok=True)
    
        
        if self.date_time is not None:
            date = self.date_time + timedelta(seconds = segment.tstart.s)
            name = f"{self.prefix}{date.strftime(self.date_format)}_{segment.dur.s:05.0f}{self.suffix}_{self.ext}.{audio_format}"
        else:
            name = f"{self.prefix}{segment.tstart.s:06.0f}_{segment.tend.s:06.0f}_{self.ext}.{audio_format}"
        return os.path.join(out_path, name)

    @staticmethod
    def export_segment_ffmpeg(
            path: str,
            out_path: str,
            ss: TimeUnit = None,
            to: TimeUnit = None,
            ss_s: float = None,
            to_s: float = None,
            overwrite = True, 
            resample: int = None,
            codec: str = None,
            **kwargs
        ) -> subprocess.CompletedProcess:
        """
        Export a single segment from TimeUnit `ss` to `to` (resp. `ss_s`, `to_s` in seconds)
        of the file provided by `path` using FFmpeg's command.
        """


        args = ["ffmpeg", "-i", path, "-loglevel", "error"]
        if ss_s is not None:
            ss = TimeUnit(ss_s)
        if ss is not None:
            args += ["-ss", str(max(0, ss.s))]
        if to_s is not None:
            to = TimeUnit(to_s)
        if to is not None:
            args += ["-to", str(to.s)]
        
        if resample is not None and isinstance(resample, int):
            args += ["-ar", str(resample)]

        if codec is not None and isinstance(codec, str):
            args += ["-c:a", codec]      



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
            resample: int = None,
            min_seg_duration: float = None,
            **kwargs
        ) -> subprocess.CompletedProcess:
        """
        Export multiple segment from TimeUnit `ss` to `to` (resp. `ss_s`, `to_s` in seconds)
        of the file provided by `path` using FFmpeg's `segment` command, considerably
        improving speed for non-overlapping (wav) segments.
        """


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
            if len(times) == 0:
                # If no times to split, add a timestamp 0 to guarantee success. TODO: Maybe make this nicer.
                times = [0]
            args += ["-segment_times", ",".join([str(t) for t in times])]
        else:
            args += ["-segment_time", str(times)]
        
        if segment_list is not None:
            args += ["-segment_list", segment_list]
        if segment_list_prefix is not None:
            args += ["-segment_list_entry_prefix", segment_list_prefix]

        
        if resample is not None and isinstance(resample, int):
            args += ["-ar", str(resample)]

        if min_seg_duration is not None and isinstance(min_seg_duration, float):
            args += ["-min_seg_duration", str(min_seg_duration)]

        args += ["-metadata", f"duration={BIRDNET_AUDIO_DURATION}"]

        args.append(out_path)

        if overwrite:
            args.append("-y")
        else: 
            args.append("-n")

        return subprocess.run(args, capture_output=True)
    
    def rename_or_delete_segments(
            self,
            base_path: str,
            label: str,
            unsegmented_fpath: str,
            segment_list: str = None,
            tstart: TimeUnit = None,
            delete_prob: float | None = None,
            **kwargs,
        ):
        """
        After using the segment command from ffmpeg (from `self.export_segments_ffmpeg()`),
        checks that all the chunks in the list provided by `segment_list` are not
        shorter than the `BIRDNET_AUDIO_DURATION`.
        If a segment is shorter, start the extraction earlier (if possible), so that
        it has the desired length.

        If a file is not wav, we also need to re-encode the segment audio file, 
        because the duration in the header is missing.

        As a result, this might take longer.

        Returns
        -------
        The number of "valid" chunks (larger than `BIRDNET_AUDIO_DURATION`) (`int`).
        """

        count = 0

        with open(segment_list, encoding='utf-8') as fp:
            csv_reader = csv.reader(fp)
            for row in csv_reader:
                tstart_f = TimeUnit(row[1])
                tend_f = TimeUnit(row[2])
                tstart_abs = tstart + tstart_f
                tend_abs = tstart + tend_f
                fpath = row[0]
                ext = fpath.split(".")[-1]
                

                if tend_f - tstart_f < BIRDNET_AUDIO_DURATION:
                    if tend_f - BIRDNET_AUDIO_DURATION > 0:
                        # If possible, start extraction earlier
                        tstart_f = tend_f - BIRDNET_AUDIO_DURATION
                        tend_f = TimeUnit(row[2])
                        tstart_abs = tstart + tstart_f
                        tend_abs = tstart + tend_f
                        seg = Segment(tstart_abs, tend_abs, label)
                        out_path = self.segment_path(base_path, seg, ext)
                        if not os.path.isfile(out_path) and (delete_prob is not None and random.random() >= delete_prob):
                            self.export_segment_ffmpeg(unsegmented_fpath, out_path, ss = tstart_f, to=tend_f)
                            count += 1
                    permission_safe(os.remove, [fpath])
                    continue

                seg = Segment(tstart_abs, tend_abs, label)
                out_path = self.segment_path(base_path, seg, ext)
                if os.path.isfile(out_path) or (delete_prob is not None and random.random() < delete_prob):
                    permission_safe(os.remove, [fpath])
                    continue

                count += 1

                if "wav" == ext.lower():
                    permission_safe(os.replace, [fpath, out_path])
                    continue

                # If the output audio is not wav, a re-encode is needed.

                self.export_segment_ffmpeg(fpath, out_path)
                permission_safe(os.remove, [fpath])
                    
        permission_safe(os.remove, [segment_list])
        return count

    
    def export_all_birdnet(
            self,
            base_path: str,
            segments: list[Segment],
            audio_format: str = "flac",
            resample: int = BIRDNET_SAMPLE_RATE,
            overlap_s: float = 0, 
            length_threshold_s = 100 * BIRDNET_AUDIO_DURATION.s,
            late_start: bool = False,
            early_stop: bool = False,
            noise_export_ratio: float = .1,
            proc_logger: ProcLogger = ProcLogger(),
            logger: Logger = Logger(),
            progress_bar: ProgressBar = None,
            **kwargs):
        """
        Export every segment to the corresponding folder in the `base_path` directory, each with the length defined 
        by the variable `BIRDNET_AUDIO_DURATION`.

        Parameters:
            - `base_path` (`str`): The base path where the exported audio clips will be saved.
            - `segments` (`list[Segment]`): A list of sound segments to export audio clips for.
            - `audio_format` (`str`, optional): The desired audio format for exported clips, as extension (default is "flac").
            - `resample` (`int`, optional): Resample output to the given value in Hz (default is `BIRDNET_SAMPLE_RATE` from `variables`).
            - `overlap_s` (`float`, optional): The amount of overlap between segments in seconds for segments longer than `BIRDNET_AUDIO_DURATION` (default is 0).
            - `length_threshold_s` (`int`, optional): Length threshold in seconds above which the algorithm will start splitting 
              the long segments using the faster ffmpeg segment command, without overlap (default is 300).
            - `late_start` (`bool`, optional): Whether to not consider the interval between the start of the recording and the first
              segment (default is False). 
            - `early_stop` (`bool`, optional): Whether to not consider the interval between the the last segment and the end of the
              recording (default is False). 
            - `logger` (`ProcLogger`): object which allow to log success and error messages from the processes.
            - `**kwargs`: Additional keyword arguments for customization.
        Example Usage:
        ```
        birdnet_instance.export_all_birdnet("./your/output/directory", segments, audio_format="wav", overlap_s=1)
        ```    
        """
        random.seed(self.basename)
        length_threshold = TimeUnit(length_threshold_s)
        segments = sorted([s.birdnet_pad() for s in segments], key=lambda seg: seg.tstart)
        annotation_start = 0 if not late_start else segments[0].tstart
        annotation_end = self.duration if not early_stop else max([s.tend for s in segments])
        duration = annotation_end - annotation_start
        overlap = TimeUnit(overlap_s)
        n_segments_original = len(segments)
        segments = [seg for seg in segments if seg.label != NOISE_LABEL]
        n_segments = len(segments)

        # for seg in segments:
        #     as_int = int(ceil(seg.tend.s))
        #     if  as_int > max_tend:
        #         max_tend = as_int

        kwargs["resample"] = resample

        # Boolean mask that for each second of the original file,
        # stores whether at least one segment overlaps or not.
        is_segment = np.zeros(int(ceil(self.duration)), np.bool_)

        for seg in segments:
            start = int(seg.tstart.s)
            end = int(ceil(seg.tend.s))
            is_segment[start:end] = True
        is_noise = ~is_segment

        diff = np.diff(is_noise.astype(np.int8), prepend=1, append=1)
        noise_tot_dur = TimeUnit(np.sum(is_noise[int(annotation_start):int(annotation_end)]))
        labelled_tot_dur = duration - noise_tot_dur
        noise_ratio = noise_tot_dur / self.duration
        noise_export_ratio = max(min(1, noise_export_ratio), 0)
        noise_export_prob = min(1, noise_ratio and noise_export_ratio / noise_ratio or 0)


        # Timestamps where the insterval changes from being 
        # segments-only to noise-only (and vice versa).
        tstamps = np.flatnonzero(np.abs(diff) == 1)

        n_labelled_chunks = 0
        basepath_noise = os.path.join(base_path, NOISE_LABEL)


        with tempfile.TemporaryDirectory() as tmpdirname:
            tmppath = lambda fname: os.path.join(tmpdirname, fname)
            listpath = tmppath(f"{self.basename}.csv")
            segment_path = tmppath(f"{self.basename}%04d.{audio_format}")

            # Start by efficiently splitting the large data file into smaller chunks of contiguous noise/segments.

            proc = self.export_segments_ffmpeg(
                path=self.path,
                out_path=segment_path,
                times = tstamps,
                segment_list=listpath,
                segment_list_prefix=tmppath(""),
                **kwargs
            )

            proc_logger.log_process(
                proc,
                f"Chunked large file into smaller ones following the timestamps: {tstamps}",
                f"Error while chunking large file into smaller ones following the timestamps {tstamps}:"
            )

            noise_chunks = 0

            seg = segments.pop(0) if segments else None
            
            with open(listpath, encoding='utf-8') as fp:
                csv_reader = csv.reader(fp)

                for row in csv_reader:
                    # For each chunk obtained

                    tstart_f = TimeUnit(row[1])
                    tend_f = TimeUnit(row[2])

                    if tend_f < annotation_start:
                        continue

                    if tstart_f > annotation_end:
                        continue

                    if (tend_f-tstart_f) < BIRDNET_AUDIO_DURATION:
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
                            if not os.path.isfile(out_path):
                                proc = self.export_segment_ffmpeg(path=fpath, ss=ss, to=to, out_path=out_path, **kwargs)

                                if proc_logger.log_process(
                                    proc,
                                    f"{seg} from file {self.path} exported to: {out_path}",
                                    f"Error while exporting {seg}:"
                                ):
                                    n_labelled_chunks += 1
                                             
                        elif (seg.dur > length_threshold or overlap == 0):
                            # If the segment is very long (above length_threshold) or overlap is 0, and the output format is wav,
                            # use the faster segment ffmpeg command.
                            # The command doesn't allow to have any overlap.

                            segment_path = self.segment_path(base_path, seg, audio_format)
                            out_dir = os.path.dirname(segment_path)
                            splits = segment_path.split(".")
                            base_out_path = ".".join(splits[:-1])

                            out_path = f"{base_out_path}_%04d.{splits[-1]}"

                            temp_seglist = os.path.join(out_dir, "list.csv")
                            seglist_prefix = os.path.join(out_dir, "")
                           
                            proc = self.export_segments_ffmpeg(
                                path=fpath,
                                ss=ss,
                                to=to, 
                                times=BIRDNET_AUDIO_DURATION.s, 
                                out_path=out_path, 
                                segment_list=temp_seglist, 
                                segment_list_prefix=seglist_prefix, 
                                **kwargs
                            )

                            proc_logger.log_process(
                                proc,
                                f"{BIRDNET_AUDIO_DURATION.s}s-segments test from {seg} in file {self.path} exported to: {out_path}",
                                f"Error while exporting {seg}:"
                            )

                            n_labelled_chunks += self.rename_or_delete_segments(base_path, seg.label, fpath, temp_seglist, tstart_f)

                        else:
                            last = False
                            n_loops = int(ceil((to - ss) / BIRDNET_AUDIO_DURATION))
                            for _ in range(n_loops):
                                to_ = ss + BIRDNET_AUDIO_DURATION
                                if to_ > to:
                                    # If it is after the end of the segment, move the start back
                                    # so that the segment has the birdnet duration and terminates 
                                    # at the end of the segment. Set the last flag to `True`, since
                                    # moving forward doesn't make any sense (the entire segment is covered).
                                    ss -= to_ - to
                                    to_ = to
                                    last = True

                                sub_seg = Segment(ss + tstart_f, to_ + tstart_f, seg.label)
                                
                                out_path = self.segment_path(base_path, sub_seg, audio_format)
                                if not os.path.isfile(out_path):
                                    proc = self.export_segment_ffmpeg(path=fpath, ss=ss, to=to, out_path=out_path, **kwargs)
                                    if proc_logger.log_process(
                                        proc,
                                        f"{sub_seg} from {seg} in file {self.path} exported to: {out_path}",
                                        f"Error while exporting {sub_seg} from {seg}:"
                                    ):
                                        n_labelled_chunks += 1

                                ss = to_ - overlap
                                if last:
                                    break
                        
                        # If we had any segment in this file chunk, 
                        # set the noise flag to False, so that this part of the 
                        # file is not identified as noise
                        noise = False

                        if progress_bar is not None:
                            progress_bar.print(1)

                        if not segments:
                            seg = None
                            break
                            
                        seg = segments.pop(0)                        
                        
                    if noise:
                        # If there are no segments in the file chunk,
                        # it can be splitted into small segments and exported
                        # as noise for training.
                        to = (tend_f - tstart_f) - ((tend_f - tstart_f) % BIRDNET_AUDIO_DURATION)
                        
                        basename_noise_split = os.path.basename(fpath).split(".")
                        os.makedirs(basepath_noise, exist_ok=True)
                        basepath_out = os.path.join(basepath_noise, ".".join(basename_noise_split[:-1]))


                        out_path = f"{basepath_out}%04d.{audio_format}"


                        temp_seglist = os.path.join(basepath_noise, "list.csv")
                        seglist_prefix = os.path.join(basepath_noise, "")

                        proc = self.export_segments_ffmpeg(
                            path=fpath,
                            to=to,
                            times=BIRDNET_AUDIO_DURATION.s,
                            out_path=out_path,
                            min_seg_duration=BIRDNET_AUDIO_DURATION.s,
                            segment_list=temp_seglist,
                            segment_list_prefix = seglist_prefix,
                            **kwargs
                        )


                        seg_noise = Segment(tstart_f, tend_f, NOISE_LABEL)
                        if proc_logger.log_process(
                            proc,
                            f"{BIRDNET_AUDIO_DURATION.s}s-segment from {seg_noise} in file {self.path} exported to: {out_path}",
                            f"Error while exporting {seg_noise} into segments:"
                        ):
                            
                            noise_chunks += self.rename_or_delete_segments(
                                base_path,
                                NOISE_LABEL,
                                fpath,
                                temp_seglist,
                                tstart_f,
                                delete_prob = (None if not noise_export_prob else 1 - noise_export_prob),
                            )

        logger.print(f"{self.path}:")    
        logger.print(
            "\t",
            f"{noise_tot_dur:.0f}s",
            "of noise,",
            f"{labelled_tot_dur:.0f}s",
            "of lables.",
            "Noise to duration ratio:",
            f"{noise_ratio:.1%}",
        )
        logger.print(
            "\t",
            n_segments_original,
            "annotated segments,",
            n_segments,
            "not blacklisted.",
            n_labelled_chunks,
            "labelled chunks,",
            noise_chunks,
            "noise chunks",
        ) 
        exported_noise_dur = noise_chunks * BIRDNET_AUDIO_DURATION
        logger.print(
            "\t",
            f"{(noise_tot_dur and exported_noise_dur/noise_tot_dur or 0):.1%}",
            "of noise exported,",f"{exported_noise_dur/self.duration:.1%}",
            "of the total duration"
        )

        # logger.print(
        #     "\t",
        #     noise_tot_dur,
        #     labelled_tot_dur,
        #     noise_ratio,
        #     noise_export_ratio,
        #     noise_export_prob,
        # )