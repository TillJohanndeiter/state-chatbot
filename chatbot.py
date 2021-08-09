import argparse
from pathlib import Path

from src.language_model import read_csv_file, LanguageModelApi
from src.graph import get_model_template, load_graph, END_STATE, START_STATE

CSV_FILE_ARG = 'samples'
GRAPH_ARG = 'graph'
MODEL_ARG = 'model'
CREATE_GRAPH_TEMPLATE = 'createGraphTemplate'

DEFAULT_GRAPH_PATH = 'graph.json'
TRAIN_ARG = 'train'
TRAINED_MODEL_FOLDER = 'trainedModels'

parser = argparse.ArgumentParser()
parser.add_argument(f'-{CSV_FILE_ARG[0]}', f'--{CSV_FILE_ARG}',
                    type=str,
                    help='csv file with sample and corresponding classes',
                    default='input_to_class.csv')
parser.add_argument(f'-{GRAPH_ARG[0]}', f'--{GRAPH_ARG}',
                    type=str,
                    help='json file that specify model with nodes and transitions',
                    default='graph.json')
parser.add_argument(f'-{MODEL_ARG[0]}', f'--{MODEL_ARG}',
                    type=str,
                    help='json file that specify model with nodes and transitions',
                    default=f'{TRAINED_MODEL_FOLDER}/model')
parser.add_argument(f'-{TRAIN_ARG[0]}', f'--{TRAIN_ARG}',
                    action='store_true',
                    help='Train model based on dataset')

parser.add_argument(f'-{CREATE_GRAPH_TEMPLATE[0]}', f'--{CREATE_GRAPH_TEMPLATE}',
                    nargs="+",
                    help='create template for model json file based on next_state names')


def write_model_template():
    state_names = START_STATE + args[CREATE_GRAPH_TEMPLATE] + END_STATE
    if len(state_names) and state_names[0].isnumeric():
        state_names = list(range(int(state_names[0])))
    state_names = [str(state_name) for state_name in state_names]
    model_json = get_model_template(state_names, all_classes)

    with open(DEFAULT_GRAPH_PATH, 'w') as file:
        file.write(model_json)


def load_output_to_class(path: Path):
    dataset = read_csv_file(path)
    all_output_classes = set(sample[1] for sample in dataset)
    output_to_cls = {cls: [] for cls in all_output_classes}

    for text, cls in dataset:
        output_to_cls[cls].append(text)

    return output_to_cls


if __name__ == '__main__':
    args = vars(parser.parse_args())

    filepath_of_csv = Path(args[CSV_FILE_ARG])
    model_path = Path(args[MODEL_ARG])
    sample_to_classes = read_csv_file(filepath_of_csv)
    all_classes = list(set(cls for sample, cls in sample_to_classes))

    if not Path(TRAINED_MODEL_FOLDER).exists():
        Path(TRAINED_MODEL_FOLDER).mkdir()

    if args[CREATE_GRAPH_TEMPLATE] is not None:
        write_model_template()
    elif args[TRAIN_ARG]:
        lng_model = LanguageModelApi()
        lng_model.train_model(filepath_of_csv)
        lng_model.save_model(model_path)
    else:
        filepath_model = args[GRAPH_ARG]
        lng_model = LanguageModelApi(model_path)

        output_to_class = load_output_to_class(Path('output_to_class.csv'))

        graph_json = load_graph(Path(filepath_model), output_to_class, lng_model)
        graph_json.start_input_loop()
