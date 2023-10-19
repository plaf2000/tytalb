from parsers import available_parsers
from base_classes import TableParser
import os


def get_parser(table_format: str, **parser_kwargs):
    table_format_name = table_format.lower()
    table_parser: TableParser = None
    for tp in available_parsers:
        tp_init: TableParser = tp(**parser_kwargs)
        if table_format_name in tp_init.names:
            table_parser = tp_init
    return table_parser

class BirdNetExtractor:
    tables_paths: list = []

    def __init__(
            self,
            table_format: str,
            tables_dir: str,
            export_dir: str,
            audio_files_dir: str,
            audio_file_ext: str,
            recursive_subfolders = True,
            **parser_kwargs
        ):

        self.parser = get_parser(table_format, **parser_kwargs)
        self.tables_dir = tables_dir
        self.audio_files_dir = audio_files_dir
        self.recursive_subfolders = recursive_subfolders
        self.export_dir = export_dir
        self.audio_file_ext = audio_file_ext
        paths = [os.path.join(self.tables_dir, f) for f in os.listdir(self.tables_dir)]
        while paths:
            p = paths.pop()
            if os.path.isfile(p) and self.parser.is_table(table_path=p):
                self.tables_paths.append(p)
            elif recursive_subfolders and os.path.isdir(p):
                paths.append(p)

    def extract_from_table(self, table_path, audio_files_dir, audio_file_ext,  **kwargs):
        detections = self.parser.get_detections(table_path, audio_files_dir, audio_file_ext)
        for det in detections:
            det.export_for_birdnet(self.export_dir, **kwargs)

    def extract_all(self, **kwargs):
        for tp in self.tables_paths:
            self.extract_from_table(tp, self.audio_files_dir, self.audio_file_ext, **kwargs)


if __name__ == "__main__":
    extr = BirdNetExtractor(
        "raven",
        "C:\\Users\\plaf\\Documents\\raven",
        "C:\\Users\\plaf\\Documents\\raven\\out",
        "C:\\Users\\plaf\\Music",
        "wav",
        False
    )

    extr.extract_all(overlap_ms=1000)
    

