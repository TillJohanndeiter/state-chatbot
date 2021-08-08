import re
from pathlib import Path


def clean_up(str_to_clean: str) -> str:
    str_to_clean = re.sub('\n', '', str_to_clean)
    str_to_clean = str_to_clean.strip()
    return str_to_clean


def read_csv_file(filepath: Path) -> [(str, str)]:
    assert filepath.exists()
    assert filepath.is_file()
    samples_to_class = []
    with open(filepath, 'r') as csv_file:
        lines = csv_file.readlines()

        for line in lines:
            line = clean_up(line)
            if line and not line.startswith('#'):
                csv_seperated = line.split(',')
                assert len(csv_seperated) == 2
                text_sample, class_of_sample = csv_seperated[0], csv_seperated[1]
                samples_to_class.append((text_sample, class_of_sample))

    return samples_to_class
