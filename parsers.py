from base_classes import TableParser, Column, DurDetection, Detection, AudioFile
from inspect import signature
import csv
import os 


def collect_args(all_locs: dict):
    args = [arg for arg in signature(TableParser.__init__).parameters]
    args.remove("self")
    args_dict = {}
    for k in all_locs.keys():
        if k in args:
            args_dict[k] = all_locs[k]
    return args_dict

class SonicParser(TableParser):
    def __init__(self, 
        names = ["sonic-visualizer", "sonic-visualiser", "sv"],
        delimiter = ",",
        tstart = Column("START", 0),
        tend = Column("END", 1),
        label = Column("LABEL", 4),
        detection_type = Detection,
        audio_file_path = None,
        header = False,
        table_fnmatch = "*.csv",
        **kwargs
    ):
        super().__init__(**collect_args(locals())) 
    

class AudacityParser(TableParser):
    def __init__(self, 
        names = ["sonic-visualizer", "sonic-visualiser", "sv"],
        delimiter = "\t",
        tstart = Column(None, 0),
        tend = Column(None, 1),
        label = Column(None, 2),
        audio_file_path = None,
        detection_type = Detection,
        header = False,
        table_fnmatch = "*.txt",
        **kwargs
    ):
        super().__init__(**collect_args(locals())) 


    


class RavenParser(TableParser): 
    def __init__(self, 
        names = ["raven", "rvn"],
        delimiter = "\t",
        tstart = Column("Begin Time (s)", 3),
        tend = Column("End Time (s)", 4),
        label = Column("Annotation", 10),
        table_fnmatch = "*.selections.txt",
        detection_type = Detection,
        **kwargs
    ):
        super().__init__(**collect_args(locals())) 



class KaleidoscopeParser(TableParser):
    def __init__(self, 
        names = ["kaleidoscope", "ks"],
        delimiter = ",",
        tstart = Column("OFFSET", 3),
        tend = Column("DURATION", 4),
        label = Column("scientific_name", 5),
        table_fnmatch = "*.csv",
        detection_type = DurDetection,
        **kwargs
    ):
        super().__init__(**collect_args(locals()))

    def __post_init__(self):
        super().__post_init__()
        self.audio_file_path: list[Column] =  [
            Column("INDIR", 0),
            Column("FOLDER", 1),
            Column("IN FILE", 2),
        ],
        self.all_columns += self.audio_file_path
    
    def get_audio_files(self, table_path, *args, **kwargs):
        with open(table_path) as fp:
            csvr = csv.reader(fp, delimiter=self.delimiter)
            if self.header:
                theader = next(csvr)
                self.set_coli(theader)
            for row in csvr:
                audio_file_path = os.path.join(*[p.get_val(row) for p in self.audio_file_path])
                if not audio_file_path in self.all_audio_files:
                    self.all_audio_files[audio_file_path] = AudioFile(audio_file_path)
                yield self.all_audio_files[audio_file_path]


class BirdNetParser(TableParser):
    def __init__(self, 
        names = ["birdnet", "bn"],
        delimiter = ",",
        tstart = Column("start_time ", 3),
        tend = Column("end_time", 4),
        label = Column("label", 6),
        table_fnmatch = "*.csv",
        detection_type = Detection,
        **kwargs
    ):
        super().__init__(**collect_args(locals())) 


available_parsers = [
    SonicParser,
    AudacityParser,
    RavenParser,
    KaleidoscopeParser,
]
