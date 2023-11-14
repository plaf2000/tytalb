import subprocess
import traceback
from typing import IO
from datetime import datetime, timedelta


class ProgressBar():
# Thanks to https://stackoverflow.com/a/37630397
    
    def __init__(self, text: str, total: int, bar_length=40):
        self.total = total
        self.bar_length = bar_length
        self.amount = 0
        self.increment=total/100
        self.text = text
        self.print(increment=0)

    def is_to_update(self, i, verbose):
                i+=1
                return verbose and not i%self.increment
    
    def print(self, increment=None):
                if increment is None:
                    increment = self.increment
                self.amount+=increment
                if self.amount > self.total:
                    self.amount = self.total

                fraction = self.amount / self.total

                arrow = int(fraction * self.bar_length - 1) * '-' + '>'
                padding = int(self.bar_length - len(arrow)) * ' '

                ending = '\n' if self.amount == self.total else '\r'

                print(f'{self.text}: [{arrow}{padding}] {int(fraction*100)}%', end=ending)

    def terminate(self):
        rest = self.total - self.amount
        if rest > 0:
            self.print(rest)

class Logger:
    def __init__(self, logfile=None, logfile_path=None, log=True, log_date=True):
        self.logfile_path=logfile_path
        self.logfile = logfile
        self.log_date = log_date
        self.log = log
        if self.logfile_path is not None:
            with open(self.logfile_path, "w", encoding='utf-8') as fp:
                fp.write("")

    def print(self, *args, **kwargs):
        if self.log:
            if self.log_date:
                args = list(args)
                args.insert(0, datetime.utcnow().isoformat()) 
            if self.logfile_path is not None:
                with open(self.logfile_path, "a", encoding='utf-8') as fp:
                    print(*args, **kwargs, file=fp)
                    return
            print(*args, *kwargs, file=self.logfile)
    
    def print_exception(self, exception):
        if self.log:
            self.print("The following error occured:")
            if self.logfile_path is not None:
                with open(self.logfile_path, "a", encoding='utf-8') as fp:
                    traceback.print_exception(exception, file=fp)
                    return
            traceback.print_exception(exception, file=fp)

        
class ProcLogger:
    def __init__(self, log_type: str = "all", logfile_success: IO = None, logfile_errors: IO = None, logfile_success_path: str = None, logfile_errors_path: str = None, **kwargs):
        self.log_errors = log_type == "errors" or log_type == "all"
        self.log_success = log_type == "all"

        self.success_logger = Logger(logfile_success, logfile_success_path, self.log_success)
        self.errors_logger = Logger(logfile_errors, logfile_errors_path, self.log_errors)
        self.errors = 0
        self.successes = 0

    def print_success(self, *args, **kwargs):
        self.success_logger.print(*args, **kwargs)

    def print_errors(self, *args, **kwargs):
        self.errors_logger.print(*args, **kwargs)


    def log_process(self, process: subprocess.CompletedProcess, success_message: str, error_message: str) -> bool:
        try:
            process.check_returncode()
            self.success_logger.print(process.args)
            self.success_logger.print(success_message)
            self.successes += 1
            return True
        except subprocess.CalledProcessError as e:
            self.errors_logger.print(error_message)
            self.errors_logger.print(f"\t{e}")
            error_lines: list[str] = process.stderr.decode('utf_8').splitlines()
            for err in error_lines:
                self.errors_logger.print(f"\t{err}")
            self.errors += 1
            return False
