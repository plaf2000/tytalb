from generic_parser import TableParser, Column, FloatColumn
from segment import DurSegment, Segment, ConfidenceSegment
from audio_file import AudioFile
from inspect import signature
import csv
import os 

"""
Custom parser to parse the table-formatted text files.
To create a custom parser, extend the `TableParser`
class and override the default arguments.

Then collect the arguments to pass to this parent
class as done in the parser list below with the method
`collect_args()`.

Use `Column` to define the index of the location (starting
 from 0) where the attribute can be retrieved.

If the tables have an header, set the class' attribute
to true and define the name of the column in the header.

Do not forget to add the newly created parser in the list of
available parsers (`available_parsers`) below.

The attribute `names` is used to identify the parser, 
make sure that there are no overlaps between different parser
in the `available_parser` list.

"""

def collect_args(all_locs: dict):
    # Do not touch this
    args = [arg for arg in signature(TableParser.__init__).parameters]
    args.remove("self")
    args_dict = {}
    for k in all_locs.keys():
        if k in args:
            args_dict[k] = all_locs[k]
    return args_dict


class SonicParser(TableParser):
    def __init__(self, 
        names = ["sonic-visualizer", "sv"],
        delimiter = ",",
        tstart = FloatColumn("START", 0),
        tend = FloatColumn("END", 1),
        label = Column("LABEL", 4),
        segment_type = Segment,
        audio_file_path = None,
        header = False,
        table_fnmatch = "*.csv",
        **kwargs
    ):
        super().__init__(**collect_args(locals())) 
    

class AudacityParser(TableParser):
    def __init__(self, 
        names = ["audacity", "ac"],
        delimiter = "\t",
        tstart = FloatColumn(colindex=0),
        tend = FloatColumn(colindex=1),
        label = Column(colindex=2),
        audio_file_path = None,
        segment_type = Segment,
        header = False,
        table_fnmatch = "*.txt",
        **kwargs
    ):
        super().__init__(**collect_args(locals())) 



class RavenParser(TableParser): 
    def __init__(self, 
        names = ["raven", "rvn"],
        delimiter = "\t",
        tstart = FloatColumn("Begin Time (s)", 3),
        tend = FloatColumn("End Time (s)", 4),
        label = Column("Annotation", 10),
        table_fnmatch = "*.selections.txt",
        segment_type = Segment,
        **kwargs
    ):
        super().__init__(**collect_args(locals())) 



class KaleidoscopeParser(TableParser):
    def __init__(self, 
        names = ["kaleidoscope", "ks"],
        delimiter = ",",
        tstart = FloatColumn("OFFSET", 3),
        tend = FloatColumn("DURATION", 4),
        label = Column("scientific_name", 5),
        table_fnmatch = "*.csv",
        segment_type = DurSegment,
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

    def get_audio_rel_no_ext_paths(self, table_path: str, tables_base_path: str):
        with open(table_path) as fp:
            csvr = csv.reader(fp, delimiter=self.delimiter)
            if self.header:
                theader = next(csvr)
                self.set_coli(theader)
            for row in csvr:
                rel_path: str = os.path.join(*[p.get_val(row) for p in self.audio_file_path[1:]])
                yield rel_path.split(".")[0]
        
    
    # def get_audio_files(self, table_path, *args, **kwargs):
    #     with open(table_path) as fp:
    #         csvr = csv.reader(fp, delimiter=self.delimiter)
    #         if self.header:
    #             theader = next(csvr)
    #             self.set_coli(theader)
    #         for row in csvr:
    #             audio_file_path = os.path.join(*[p.get_val(row) for p in self.audio_file_path])
    #             self.all_audio_files.setdefault(audio_file_path, AudioFile(audio_file_path))
    #             yield self.all_audio_files[audio_file_path]



class BirdNetRavenParser(RavenParser):
    def __init__(self, 
        names = ["birdnet_raven", "bnrv"],
        delimiter = "\t",
        tstart = FloatColumn("Begin Time (s)", 3),
        tend = FloatColumn("End Time (s)", 4),
        label = Column("Species Code", 10),
        table_fnmatch = "*.BirdNET.selection.table.txt",
        segment_type = ConfidenceSegment,
        **kwargs
    ):
        super().__init__(**collect_args(locals())) 
    

    def __post_init__(self):
        super().__post_init__()
        self.confidence = FloatColumn("Confidence", 11)
        self.columns: list[Column] = [self.tstart, self.tend, self.label, self.confidence]


class BirdNetCSVParser(TableParser):
    def __init__(self, 
        names = ["birdnet", "bn"],
        delimiter = ",",
        tstart = FloatColumn("start_time", 3),
        tend = FloatColumn("end_time", 4),
        label = Column("label", 6),
        table_fnmatch = "*.csv",
        segment_type = ConfidenceSegment,
        **kwargs
    ):
        super().__init__(**collect_args(locals()))
    
    def __post_init__(self):
        super().__post_init__()
        self.confidence = FloatColumn("Confidence", 11)
        self.columns: list[Column] = [self.tstart, self.tend, self.label, self.confidence]


available_parsers = [
    SonicParser,
    AudacityParser,
    RavenParser,
    KaleidoscopeParser,
    BirdNetRavenParser,
]


ap_names = []
for ap in available_parsers:
    ap_names += ap().names

def get_parser(table_format: str, **parser_kwargs):
    table_format_name = table_format.lower()
    table_parser: TableParser = None
    for tp in available_parsers:
        tp_init: TableParser = tp(**parser_kwargs)
        if table_format_name in tp_init.names:
            table_parser = tp_init
    return table_parser