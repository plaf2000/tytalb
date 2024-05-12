from typing import Union

class TimeUnit(float):
    def __new__(cls, s: Union[float, str] = 0, ms: Union[float, str, None] = None):
        if ms is not None:
            s = float(ms) / 1000
        return super().__new__(cls, float(s))

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
