from units import TimeUnit
from variables import BIRDNET_AUDIO_DURATION
import  intervaltree as it


class Segment(it.Interval):
    """
    Extends the `intervaltree.Interval` class. Uses `bytearray` to represent the 
    label string in the data, in order to make it mutable.
    """
    tstart: TimeUnit
    tend: TimeUnit
    label: str | None

    def __new__(cls, tstart_s: float, tend_s: float, label: str | bytearray = ""):
        if isinstance(label, str):
            label = label.encode()
        if label is None:
            label = b""
        return super(Segment, cls).__new__(cls, tstart_s, tend_s, bytearray(label))

    @property
    def tstart(self):
        return TimeUnit(self.begin)
    
    @property
    def tend(self):
        return TimeUnit(self.end)

    @property
    def label(self):
        return self.data.decode()
    
    @label.setter
    def label(self, l: str):
        self.data[:] = l.encode()
        
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

    def overlapping_time(self, other: 'Segment'):
        # Copied from Pshemek
        ov_b = max(self.tstart, other.tstart)
        ov_e = min(self.tend, other.tend)
        return max(0.0, ov_e - ov_b)

    def overlapping_perc(self, other: 'Segment'):
        return self.overlapping_time(other)/self.dur
    

    def __str__(self):
        seg_name = f" \"{self.label}\"" if self.label is not None else ""
        return f"Segment{seg_name}: [{self.tstart.time_str(True)}, {self.tend.time_str(True)}]"
    
    @staticmethod
    def from_interval(interval: it.Interval):
        """
        Create a segment from an a treeinterval Interval.
        """
        return Segment(interval.begin, interval.end, interval.data)

    @staticmethod
    def get_intervaltree(segments: list['Segment']):
        """
        Returns the IntervalTree datastructure from a list of segments.
        """
        return it.IntervalTree(segments)

    

class DurSegment(Segment):
    def __init__(self, tstart, dur, label):
        super().__init__(tstart, tstart+dur, label)    









