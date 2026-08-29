from announcement_body_v3 import runner as parent
from .binder import extract_document


def run():
    original = parent.extract_document
    original_directory, original_verify = parent.DIRECTORY, parent.verify
    from .freeze import DIRECTORY, verify
    try:
        parent.extract_document = extract_document
        parent.DIRECTORY = DIRECTORY
        parent.verify = verify
        return parent.run()
    finally:
        parent.extract_document = original
        parent.DIRECTORY = original_directory
        parent.verify = original_verify

