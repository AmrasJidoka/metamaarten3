from werkzeug.utils import secure_filename
from os import path, remove

class Document(object):
    def __init__(self, request_file):
        self.request_file = request_file

    def __enter__(self):
        self.filename = path.join("documents", secure_filename(self.request_file.filename))
        self.request_file.save(self.filename)
        return self

    def __exit__(self, *args):
        remove(self.filename)