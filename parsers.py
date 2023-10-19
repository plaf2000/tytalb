from base_classes import TableParser, Column, DurDetection, Detection
from inspect import signature

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
        file_ext = "csv",
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
        file_ext = "txt",
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
        file_ext = "selections.txt",
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
        file =  [
            Column("INDIR", 0),
            Column("FOLDER", 1),
            Column("IN FILE", 2),
        ],
        detection_type = DurDetection,
        **kwargs
    ):
        super().__init__(**collect_args(locals())) 


available_parsers = [
    SonicParser,
    AudacityParser,
    RavenParser,
    KaleidoscopeParser,
]
