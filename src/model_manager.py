import json
from pathlib import Path

CLASS_TO_TRANSITIONS = 'classToTransitions'


def get_model_template(state_names: [str], all_classes: [str]) -> str:
    model_as_dict = {}
    for state in state_names:
        model_as_dict[state] = {CLASS_TO_TRANSITIONS: {cls: 'GREET_USER' for cls in all_classes}}

    return json.dumps(model_as_dict, indent=4)


def load_model(model_path: Path):
    assert model_path.exists()
    assert model_path.is_file()

    with open(model_path, 'r') as model_file:
        model_json = json.load(model_file)

    return model_json


