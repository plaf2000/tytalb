from segment import Segment, ConfidenceSegment, ConfidenceFreqSegment
import csv
from parsers import collect_args
from functools import partial
import os

class TableWriter:
    
    def __init__(self, fpath: str, header: list[str] = None,*writer_args, **writer_kwargs):
        self.header = header
        self.header_written = False
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        self.fpath = fpath
        self.get_writer = partial(csv.writer, *writer_args, **writer_kwargs)


    def write_segment(self, segment: Segment, confidence: float | None = None):
        row = [segment.tstart, segment.tend, segment.label]
        if confidence is not None:
            row.append(confidence)
        self.writer.writerow(row)

    def write_header(self):
        if self.header is not None and not self.header_written:
            self.writer.writerow(self.header)
            self.header_written = True

    def write_segments_and_confidence(self, segments: list[ConfidenceSegment]):
        with open(self.fpath, "w", newline='') as fp:
            self.writer = self.get_writer(fp)
            self.write_header()

            for seg in segments:
                self.write_segment(seg, seg.confidence)


class RavenWriter(TableWriter):
    def __init__(self, fpath:str, header = [
        "Selection",
        "View",
        "Channel",
        "Begin File",
        "Begin Time (s)",
        "End Time (s)",
        "Low Freq (Hz)",
        "High Freq (Hz)",
        "Species Code",
        "Common Name",
        "Confidence"
    ], fmax = 15000):
        super().__init__(fpath, header, delimiter="\t")
        self.fmax = fmax
        self.sel_i = 1
    
    def write_segment(self, segment: ConfidenceFreqSegment, confidence: float):
        row = ([
            self.sel_i,
            "Spectrogram 1",
            "1",
            "Begin File",
            segment.tstart,
            segment.tend,
            segment.fstart,
            segment.fend,
            segment.label,
            str(segment.label).split("_")[1],
            confidence
        ])
        self.writer.writerow(row)
        self.sel_i += 1
    
