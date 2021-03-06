'''
Script to start or train the chatbot.
'''
import argparse
import csv
import logging as log
import sys
from pathlib import Path
from src.graph import get_model_template, load_graph, END_STATE, START_STATE

INPUT_DS_CSV = 'inputDs'
OUTPUT_DS_CSV = 'outputDs'
GRAPH_ARG = 'graph'
MODEL_NAME_ARG = 'model'
CREATE_GRAPH_ARG = 'createGraphTemplate'
TRAIN_ARG = 'train'
VERBOSITY_ARG = 'verbose'
DEFAULT_GRAPH_FILENAME = 'graph.json'
OUTPUT_TO_CLASS_FILENAME = 'output_to_class.csv'
INPUT_TO_CLASS_FILENAME = 'input_to_class.csv'
TRAINED_MODEL_FOLDER = 'trainedModels'

parser = argparse.ArgumentParser()
parser.add_argument(f'-{INPUT_DS_CSV[0]}', f'--{INPUT_DS_CSV}',
                    type=str,
                    help='Csv file with user input and labeled classes',
                    default=INPUT_TO_CLASS_FILENAME)
parser.add_argument(f'-{VERBOSITY_ARG[0]}', f'--{VERBOSITY_ARG}',
                    help='Csv file with user input and labeled classes',
                    action='store_true')
parser.add_argument(f'-{GRAPH_ARG[0]}', f'--{GRAPH_ARG}',
                    type=str,
                    help='json file that specify model with nodes and transitions',
                    default=DEFAULT_GRAPH_FILENAME)
parser.add_argument(f'-{MODEL_NAME_ARG[0]}', f'--{MODEL_NAME_ARG}',
                    type=str,
                    help='Folder of used language model',
                    default=f'{TRAINED_MODEL_FOLDER}/model')
parser.add_argument(f'-{TRAIN_ARG[0]}', f'--{TRAIN_ARG}',
                    action='store_true',
                    help='Train model based on dataset')

parser.add_argument(f'-{CREATE_GRAPH_ARG[0]}', f'--{CREATE_GRAPH_ARG}',
                    nargs="+",
                    help='Create template for model json file')


def read_csv_file(filepath: Path) -> [(str, str)]:
    '''
    Read the csv file at filepath.
    :param filepath: filepath of csv file
    :return: list of parsed tuples (input/output, class)
    '''
    assert filepath.exists()
    assert filepath.is_file()
    samples_to_class = []

    with open(filepath, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        for row in reader:
            # No comment or empty line
            if len(row) > 0 and not row[0].startswith('#'):
                text_sample, class_of_sample = row[0].strip(), row[1].strip()
                samples_to_class.append((text_sample, class_of_sample))

    return samples_to_class


def write_graph_template():
    '''
    Write .json template for graph to filepath passed by GRAPH_ARG argument
    :return: None
    '''
    state_names = [START_STATE] + args[CREATE_GRAPH_ARG] + [END_STATE]
    if len(state_names) and state_names[0].isnumeric():
        state_names = list(range(int(state_names[0])))
    state_names = [str(state_name) for state_name in state_names]
    model_json = get_model_template(state_names, all_classes)

    with open(args[GRAPH_ARG], 'w') as file:
        file.write(model_json)


def load_output_to_class(path: Path):
    '''
    Load dictionary that map output to classes/label
    :param path: path of output csv file
    :return: output to class dictionary
    '''
    dataset = read_csv_file(path)
    all_output_classes = set(sample[1] for sample in dataset)
    output_to_cls = {cls: [] for cls in all_output_classes}

    for text, cls in dataset:
        output_to_cls[cls].append(text)

    return output_to_cls


if __name__ == '__main__':
    args = vars(parser.parse_args())

    filepath_of_input_csv = Path(args[INPUT_DS_CSV])
    model_path = Path(args[MODEL_NAME_ARG])
    input_to_classes = read_csv_file(filepath_of_input_csv)
    all_classes = list(set(cls for sample, cls in input_to_classes))

    if not Path(TRAINED_MODEL_FOLDER).exists():
        Path(TRAINED_MODEL_FOLDER).mkdir()

    if args[VERBOSITY_ARG]:
        log.StreamHandler(stream=sys.stdout)
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    if args[CREATE_GRAPH_ARG] is not None:
        write_graph_template()
    elif args[TRAIN_ARG]:
        from src.language_model import LanguageModelApi

        lng_model = LanguageModelApi()
        lng_model.train_model(input_to_classes)
        lng_model.save_model(model_path)
    else:
        from src.language_model import LanguageModelApi

        filepath_model = args[GRAPH_ARG]
        lng_model = LanguageModelApi(model_path)

        output_to_class = load_output_to_class(Path(OUTPUT_TO_CLASS_FILENAME))

        graph_json = load_graph(Path(filepath_model), output_to_class, lng_model)
        graph_json.start_input_loop()
