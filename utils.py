import json


def read_json_lines(path, n_lines=None):
    """Creates a generator which reads and returns lines of
    a json lines file, one line at a time, each as a dictionary.

    This could be used as a memory-efficient alternative of `pandas.read_json`
    for reading a json lines file.
    """
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if n_lines == i:
                break
            yield json.loads(line)


class JSONLinesWriter:
    """
    Helper class to write list of dictionaries into a file in json lines
    format, i.e. one json record per line.
    """

    def __init__(self, file_path):
        self.fd = None
        self.file_path = file_path
        self.delimiter = "\n"

    def open(self):
        self.fd = open(self.file_path, "w")
        self.first_record_written = False
        return self

    def close(self):
        self.fd.close()
        self.fd = None

    def write_record(self, obj):
        if self.first_record_written:
            self.fd.write(self.delimiter)
        self.fd.write(json.dumps(obj))
        self.first_record_written = True

    def __enter__(self):
        return self.open()

    def __exit__(self, type, value, traceback):
        self.close()
