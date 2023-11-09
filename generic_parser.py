import fnmatch
import os
import csv
from typing import Generator, IO, Callable, Any
from dataclasses import dataclass
from segment import Segment
from audio_file import AudioFile

@dataclass
class Column:
    colname: str = None
    colindex: int = None
    

    def set_coli(self, header_row: list):
        """
        Finds the column in the header.
        """
        self.colindex = header_row.index(self.colname)

    def get_val(self, row: list):
        """
        Given the row's cells as list, return the value of the corresponding column.
        """
        return self.read_func(row[self.colindex])

    def read_func(self, cell: str):
        return cell

class IntColumn(Column):
    def read_func(self, cell:str):
        return int(cell)
    
class FloatColumn(Column):
    def read_func(self, cell:str):
        return float(cell)


@dataclass
class TableParser:
    names: list[str]
    delimiter: str
    tstart: Column
    tend: Column
    label: Column
    segment_type: Segment
    header: bool = True
    table_fnmatch: str = "*.csv"

    def __post_init__(self):
        # `self.columns` lists the columns used for retrieving the segment data (order is relevant!)
        self.columns: list[Column] = [self.tstart, self.tend, self.label]
        # `self.all_columns` lists all the columns, it is used to set the indices from the header
        self.all_columns: list[Column] = self.columns

        # Dictionary mapping from paths to `AudioFile` objects contained in the segment table
        # (usually, only one).
        self.all_audio_files: dict[str, AudioFile] = {}

    def set_coli(self, *args, **kwargs):
        """
        Set the column indices for columns that have headers.
        """
        if self.header:
            for col in self.all_columns:
                col.set_coli(*args, **kwargs)

    def get_segment(self, row: list) -> Segment:
        """
        Instantiate the `Segment` object by reading the row values.
        """
        return self.segment_type(
            *[col.get_val(row) for col in self.columns]
        )

    def get_segments(self, table_path: str, *args, **kwargs) -> Generator[Segment, None, None]:
        """
        Returns a generator that for each line of the table yields the segment.
        If the table has an header, it first sets the columns using the header.
        """
        with open(table_path) as fp:
            csvr = csv.reader(fp, delimiter=self.delimiter)
            if self.header:
                theader = next(csvr)
                self.set_coli(theader)
            for row in csvr:
                yield self.get_segment(row)

    def get_audio_rel_no_ext_paths(self, table_path: str, tables_base_path: str):
        """
        Given the table path, the directory containing the audio file and the audio file
        extenstion, returns a generator that yields the path to theaudio file corresponding 
        to each segment.
        The path is relative to the `tables_base_path` and doesn't contain the audio file 
        extension. 
        This is used to uniquely identify detections, i.e. two detections have a different 
        return value iff. they are from two different files, but the path doesn't need to exist.

        By default, there is only one audio file per table, which is retrieved
        by looking in the audio directory for audio files that have the same 
        name as the table + the provided extension in the arguments.
        """
        table_basename = os.path.basename(table_path)
        table_subpath = os.path.dirname(table_path)[len(tables_base_path):]
        audio_rel_no_ext_paths = os.path.join(table_subpath, table_basename.split(".")[0])
        with open(table_path) as fp:
            csvr = csv.reader(fp, delimiter=self.delimiter)
            for _ in csvr:
                yield audio_rel_no_ext_paths




            

    def is_table(self, table_path: str) -> bool:
        """
        Returns if the provided path matches the tables' file name pattern.
        """
        basename = os.path.basename(table_path)
        return fnmatch.fnmatch(basename, self.table_fnmatch)