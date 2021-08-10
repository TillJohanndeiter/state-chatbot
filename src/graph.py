import json
from pathlib import Path
from random import choice

CLASS_TO_TRANSITIONS = 'classToTransitions'
SAY_ON_ENTRY = 'sayOnEntry'
SAY_ON_EXIT = 'sayOnExit'
NEXT_STATE = 'nextState'

START_STATE = "START"
END_STATE = "END"


def get_model_template(state_names: [str], all_classes: [str]) -> str:
    model_as_dict = {}
    for state in state_names:
        model_as_dict[state] = {CLASS_TO_TRANSITIONS: {cls: None for cls in all_classes},
                                SAY_ON_ENTRY: None, SAY_ON_EXIT: None, NEXT_STATE: None}

    return json.dumps(model_as_dict, indent=4)


class State:

    def __init__(self, name, language_model, say_on_entry=None, say_on_exit=None, next_state=None):
        self.name = name
        self.language_model = language_model
        self.__cls_to_transition = None
        self.say_on_entry = say_on_entry
        self.say_on_exit = say_on_exit
        self.__next_state = next_state

    def do_transition(self, graph):
        assert self.__cls_to_transition is not None

        if self.say_on_entry is not None:
            print(choice(self.say_on_entry))
        self.on_entry()

        if self.__next_state is None:
            user_input = input()
            cls = self.language_model.classify_sentence(user_input, self.__cls_to_transition.keys())
            print(f'Classified as {cls}')
            next_state = self.__cls_to_transition[cls]
        else:
            next_state = self.__next_state

        self.on_exit()

        if self.say_on_exit is not None:
            print(choice(self.say_on_exit))

        graph.set_state(next_state)

        if next_state.name != END_STATE:
            next_state.do_transition(graph)

    def on_exit(self):
        pass

    def on_entry(self):
        pass

    def set_cls_to_transition(self, cls_to_transition: dict):
        self.__cls_to_transition = cls_to_transition

    def get_cls_to_transition(self):
        return self.__cls_to_transition

    def set_next_state(self, next_state):
        self.__next_state = next_state

    def __str__(self):
        return self.name


class MakeCoffee(State):

    def on_entry(self):
        print('Callback to rEallY make Coffe :)')


class Graph:

    def __init__(self, init_state: State):
        self.__current_state = init_state

    def set_state(self, state: State):
        print(f'Next state {state}')
        self.__current_state = state

    def start_input_loop(self):
        self.__current_state.do_transition(self)


def load_graph(model_path: Path, class_to_output: dict, language_model) -> Graph:
    assert model_path.exists()
    assert model_path.is_file()

    with open(model_path, 'r') as model_file:
        model_json = json.load(model_file)

        name_to_node = {}

        for state_name in model_json:
            say_on_entry = None
            say_on_exit = None

            state_json = model_json[state_name]

            if SAY_ON_ENTRY in state_json and state_json[SAY_ON_ENTRY] is not None:
                say_on_entry = class_to_output[state_json[SAY_ON_ENTRY]]

            if SAY_ON_EXIT in state_json and state_json[SAY_ON_EXIT] is not None:
                say_on_exit = class_to_output[state_json[SAY_ON_EXIT]]

            all_subclasses = [cls for cls in State.__subclasses__() if
                              cls.__name__.lower() == state_name.replace('_', '').lower()]

            if len(all_subclasses) > 0:
                name_to_node[state_name] = all_subclasses[0](state_name, language_model,
                                                             say_on_entry=say_on_entry,
                                                             say_on_exit=say_on_exit)
            else:
                name_to_node[state_name] = State(state_name, language_model,
                                                 say_on_entry=say_on_entry,
                                                 say_on_exit=say_on_exit)

        for state_name in model_json:
            state = name_to_node[state_name]
            state.set_cls_to_transition(
                {cls: name_to_node[name] for cls, name in
                 model_json[state_name][CLASS_TO_TRANSITIONS].items() if name is not None})

            if NEXT_STATE in model_json[state_name] and model_json[state_name][
                NEXT_STATE] is not None:
                state.set_next_state(name_to_node[model_json[state_name][NEXT_STATE]])

    init_state = name_to_node[START_STATE]

    return Graph(init_state)
