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
        try:
            self.colindex = header_row.index(self.colname)
        except ValueError as e:
            raise ValueError(f"'{self.colname}' not found in the header row {header_row}: {e}")
    def get_val(self, row: list):
        """
        Given the row's cells as list, return the value of the corresponding column.
        """
        if len(row) <= self.colindex:
            raise IndexError(f"Row {row} is too short.")
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
    table_per_file: bool = True

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

    def get_segment(self, row: list, line_number: int) -> Segment:
        """
        Instantiate the `Segment` object by reading the row values.
        """
        try:
            return self.segment_type(
                *[col.get_val(row) for col in self.columns], line_number
            )
        except ValueError or IndexError as e:
            raise ValueError(f"{self.names[0]} parser unable to read row {row}: {e}")
    
    def valid_format(self, table_path: str, *args, **kwargs) -> bool:
        if not self.is_table(table_path):
            return False
        try:
            list(zip(range(2), self.get_segments(table_path)))
        except ValueError as e:
            return False
        return True

    def get_segments(self, table_path: str, skip_empty_row=True, *args, **kwargs) -> Generator[Segment, None, None]:
        """
        Returns a generator that for each line of the table yields the segment.
        If the table has an header, it first sets the columns using the header.
        """
        with open(table_path, encoding='utf-8') as fp:
            csvr = csv.reader(fp, delimiter=self.delimiter)
            line_offset = 1
            if self.header:
                theader = next(csvr)
                self.set_coli(theader)
                line_offset = 2

            for i, row in enumerate(csvr):
                try:
                    if skip_empty_row and (len(row)==0 or (len(row)==1 and row[0].strip()=='')):
                        print(f"Warning, empty row {row} skipped")
                        continue
                    yield self.get_segment(row, i + line_offset)
                except ValueError as e:
                    raise ValueError(f"Exception on row {i}: {e}")

    def get_audio_rel_no_ext_path(self, table_path: str, tables_base_path: str):
        table_basename = os.path.basename(table_path)
        table_subpath = os.path.dirname(table_path)
        table_subpath = table_subpath[len(tables_base_path):]
        
        while table_subpath.startswith(os.sep):
            table_subpath = table_subpath[1:]
        audio_rel_no_ext_paths = os.path.join(table_subpath, table_basename.split(".")[0])

        return audio_rel_no_ext_paths

    def get_audio_rel_no_ext_paths(self, table_path: str, tables_base_path: str):
        """
        Given the table path, the directory containing the audio file and the audio file
        extenstion, returns a generator that yields the path to the audio file corresponding 
        to each segment.
        The path is relative to the `tables_base_path` and doesn't contain the audio file 
        extension. 
        This is used to uniquely identify detections, i.e. two detections have a different 
        return value iff. they are from two different files, but the path doesn't need to exist.

        By default, there is only one audio file per table, which is retrieved
        by looking in the audio directory for audio files that have the same 
        name as the table + the provided extension in the arguments.
        """
        audio_rel_no_ext_paths = self.get_audio_rel_no_ext_path(table_path, tables_base_path)
        with open(table_path, encoding='utf-8') as fp:
            csvr = csv.reader(fp, delimiter=self.delimiter)
            for _ in csvr:
                yield audio_rel_no_ext_paths


    def is_table(self, table_path: str) -> bool:
        """
        Returns if the provided path matches the tables' file name pattern.
        """
        basename = os.path.basename(table_path)
        return fnmatch.fnmatch(basename, self.table_fnmatch)
    
    def is_table_per_file(self, table_path: str) -> bool:
        return self.table_per_file