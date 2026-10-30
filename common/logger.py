import sys


class Logger:
    def __init__(self, filename: str, mode: str = 'w'):
        self.console = sys.stdout
        self.file = open(filename, mode)

    def write(self, message: str):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()
