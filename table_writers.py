from segment import Segment 
import csv
from parsers import collect_args
from functools import partial

class TableWriter:
    
    def __init__(self, header: list[str] = None, *writer_args, **writer_kwargs):
        self.header = header
        self.header_written = False
        self.writer = csv.writer(*writer_args, **writer_kwargs)

    def write_segment(self, segment: Segment, confidence: float | None = None):
        row = [segment.tstart, segment.tend, segment.label]
        if confidence is not None:
            row.append(confidence)
        self.writer.writerow(row)

    def write_header(self):
        if self.header is not None and not self.header_written:
            self.writer.writerow(self.header)
            self.header_written = True
            

    def write_segments(self, segments: list[Segment]):
        self.write_header()
        for seg in segments:
            self.write_segment(seg)

class RavenWriter(TableWriter):
    def __init__(self, header = [
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
        super().__init__(header, delimiter="\t")
        self.fmax = fmax
    
    def write_segment(self, segment: Segment, confidence: float):
        self.writer.writerow([
        "Selection",
        "Spectrogram 1",
        "1",
        "Begin File",
        segment.tstart,
        segment.tend,
        0,
        self.fmax,
        segment.label,
        str(segment.label).split("_")[1],
        confidence
    ])