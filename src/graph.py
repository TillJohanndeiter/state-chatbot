'''
Model for interaction graph.
'''
import json
import logging as log
from pathlib import Path
from random import choice

CLASS_TO_TRANSITIONS = 'classToTransitions'
SAY_ON_ENTRY = 'sayOnEntry'
SAY_ON_EXIT = 'sayOnExit'
NEXT_STATE = 'nextState'

START_STATE = "START"
END_STATE = "END"


def get_model_template(state_names: [str], all_classes: [str]) -> str:
    '''
    generate json template for graph model.
    :param state_names: name of all states
    :param all_classes: all labels of user inputs
    :return: json template
    '''

    model_as_dict = {}
    for state in state_names:
        model_as_dict[state] = {CLASS_TO_TRANSITIONS: {cls: None for cls in all_classes},
                                SAY_ON_ENTRY: None, SAY_ON_EXIT: None, NEXT_STATE: None}

    return json.dumps(model_as_dict, indent=4)


class State:
    '''
    State/Node in the conversation graph.
    '''

    def __init__(self, name, language_model, say_on_entry=None, say_on_exit=None):
        self.name = name
        self.__language_model = language_model
        self.__cls_to_transition = None
        self.__next_state = None
        self.__say_on_entry = say_on_entry
        self.__say_on_exit = say_on_exit

    def do_transition(self, graph):
        '''
        Execute transition and set next state to graph
        :param graph: transition graph of the state
        :return: None
        '''
        assert self.__cls_to_transition is not None \
               or self.__next_state is not None or self.name == END_STATE

        if self.__say_on_entry is not None:
            print(choice(self.__say_on_entry))
        self.on_entry()

        if self.__next_state is None and self.name != END_STATE:
            user_input = input()
            cls = self.__language_model.classify_sentence(user_input,
                                                          self.__cls_to_transition.keys())
            log.info('"%s" classified as %s', user_input, cls)
            next_state = self.__cls_to_transition[cls]
        else:
            next_state = self.__next_state

        self.on_exit()

        if self.__say_on_exit is not None:
            print(choice(self.__say_on_exit))

        graph.set_state(next_state)

    def on_exit(self):
        '''
        Callback if state is left
        :return: None
        '''

    def on_entry(self):
        '''
        Callback if state is reached
        :return: None
        '''

    def set_cls_to_transition(self, cls_to_transition: dict):
        self.__cls_to_transition = cls_to_transition

    def get_cls_to_transition(self):
        return self.__cls_to_transition

    def set_next_state(self, next_state):
        self.__next_state = next_state

    def __str__(self):
        return self.name


class Graph:
    '''
    Represents conversation flow.
    '''

    def __init__(self, init_state: State):
        self.__current_state = init_state

    def set_state(self, state: State):
        log.info('Set next state: %s', state)
        self.__current_state = state

    def start_input_loop(self):
        while self.__current_state.name != END_STATE:
            self.__current_state.do_transition(self)


def load_graph(model_path: Path, class_to_output: dict, language_model) -> Graph:
    '''
    load graph from json file and set up graph with node.
    :param model_path: path of json file
    :param class_to_output:
    :param language_model: fine tuned language model
    :return: Created Graph from json file
    '''

    assert model_path.exists()
    assert model_path.is_file()

    with open(model_path, 'r') as model_file:
        model_json = json.load(model_file)

        name_to_node = {}
        # pylint: disable=unused-import disable=import-outside-toplevel disable=cyclic-import
        from src.coffee import MakeCoffee, MakeTee, MakeChocolate, VerifyOrder

        for state_name in model_json:
            _load_node(class_to_output, language_model, model_json, name_to_node, state_name)

        for state_name in model_json:
            _set_transitions_to_node(model_json, name_to_node, state_name)

        init_state = name_to_node[START_STATE]
        return Graph(init_state)


def _set_transitions_to_node(model_json, name_to_node, state_name):
    state = name_to_node[state_name]
    state_json = model_json[state_name]
    if CLASS_TO_TRANSITIONS in state_json:
        state.set_cls_to_transition(
            {cls: name_to_node[name] for cls, name in
             state_json[CLASS_TO_TRANSITIONS].items() if name is not None})
    if NEXT_STATE in state_json and state_json[NEXT_STATE] is not None:
        state.set_next_state(name_to_node[state_json[NEXT_STATE]])


def _load_node(class_to_output, language_model, model_json, name_to_node, state_name):
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
