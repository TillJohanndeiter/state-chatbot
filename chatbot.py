import argparse
from pathlib import Path

from src.csv_parser import read_csv_file
from model_manager import get_model_template, load_model

CSV_FILE_ARG = 'samples'
GRAPH_ARG = 'graph'
CREATE_GRAPH_TEMPLATE = 'createGraphTemplate'

DEFAULT_GRAPH_PATH = 'graph.json'

parser = argparse.ArgumentParser()
parser.add_argument(f'-{CSV_FILE_ARG[0]}', f'--{CSV_FILE_ARG}',
                    type=str,
                    help='csv file with sample and corresponding classes',
                    default='samples_to_class.csv')
parser.add_argument(f'-{GRAPH_ARG[0]}', f'--{GRAPH_ARG}',
                    type=str,
                    help='json file that specify model with nodes and transitions',
                    default='model.json')
parser.add_argument(f'-{CREATE_GRAPH_TEMPLATE[0]}', f'--{CREATE_GRAPH_TEMPLATE}',
                    nargs="+",
                    help='create template for model json file based on state names')


def write_model_template():
    state_names = args[CREATE_GRAPH_TEMPLATE]
    if len(state_names) and state_names[0].isnumeric():
        state_names = list(range(int(state_names[0])))
    state_names = [str(state_name) for state_name in state_names]
    model_json = get_model_template(state_names, all_classes)

    with open(DEFAULT_GRAPH_PATH, 'w') as file:
        file.write(model_json)


if __name__ == '__main__':
    args = vars(parser.parse_args())

    filepath_of_csv = args[CSV_FILE_ARG]
    sample_to_classes = read_csv_file(Path(filepath_of_csv))
    all_classes = list(set(cls for sample, cls in sample_to_classes))

    if args[CREATE_GRAPH_TEMPLATE] is not None:
        write_model_template()
    else:
        filepath_model = args[GRAPH_ARG]
        model_json = load_model(Path(filepath_model))
