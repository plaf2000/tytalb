from base_classes import TableParser, Column, DurDetection

class SonicParser(TableParser):
    names = ["sonic-visualizer", "sonic-visualiser", "sv"]
    delimiter = ","
    tstart = Column("START", 0)
    tend = Column("END", 1)
    label = Column("LABEL", 4)
    header = False

class AudacityParser(TableParser):
    names = ["sonic-visualizer", "sonic-visualiser", "sv"]
    delimiter = "\t"
    tstart = Column(None, 0)
    tend = Column(None, 1)
    label = Column(None, 2)
    header = False
    file_ext = "txt"


class RavenParser(TableParser):
    names = ["raven", "rvn"]
    delimiter = "\t"
    tstart = Column("Begin Time (s)", 3)
    tend = Column("End Time (s)", 4)
    label = Column("Annotation", 10)
    file_ext = "selections.txt"

class KaleidoscopeParser(TableParser):
    names = ["kaleidoscope", "ks"]
    delimiter = ","
    tstart = Column("OFFSET", 3)
    tend = Column("DURATION", 4)
    label = Column("scientific_name", 5)
    file =  [
        Column("INDIR", 0),
        Column("FOLDER", 1),
        Column("IN FILE", 2),
    ]
    detection_type = DurDetection


available_parsers = [
    SonicParser,
    AudacityParser,
    RavenParser,
    KaleidoscopeParser,
]