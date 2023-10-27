from units import TimeUnit
from variables import BIRDNET_AUDIO_DURATION

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

    def overlapping_time(self, other: 'Segment'):
        # Copied from Pshemek
        ov_b = max(self.tstart, other.tstart)
        ov_e = min(self.tend, other.tend)
        return max(0.0, ov_e - ov_b)

    def overlapping_perc(self, other: 'Segment'):
        return self.overlapping_time(other)/self.dur
    
    def overlaps(self, other: 'Segment'):
        return self.overlapping_time(other) > 0

    def __str__(self):
        seg_name = f" \"{self.label}\"" if self.label is not None else ""
        return f"Segment{seg_name}: [{self.tstart.time_str(True)}, {self.tend.time_str(True)}]"
    
    

class DurSegment(Segment):
    def __init__(self, tstart, dur, label):
        super().__init__(tstart, tstart+dur, label)    









