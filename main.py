import argparse
from pathlib import Path

from src.csv_parser import read_csv_file

CSV_FILE_ARG = 'samples'

parser = argparse.ArgumentParser()
parser.add_argument('-s', f'--{CSV_FILE_ARG}',
                    type=str,
                    help='csv file with sample and corresponding classes',
                    default='samples_to_class.csv')

if __name__ == '__main__':
    args = vars(parser.parse_args())
    filepath_of_csv = args[CSV_FILE_ARG]

    sample_to_classes = read_csv_file(Path(filepath_of_csv))
    print(sample_to_classes)
