# State based Chatbot

Framework for set up a state based chatbot. The conversation flow is represented by a graph. The graph is configurable by editing json and csv files and do not require any programming knowledge. Furthermore, the framework contains interfaces to set callbacks in the conversation flow. The transformer bert is used as language model and fine-tuned to labelled user input samples.

The project contains a dummy coffee machine to show how the framework can be used in combination with other components. 

# Installation

Supported python version: 3.9

We recommend using a virtual environment especially to avoid dependency problems. 

Create virutal environment:

```
python -m pip install --user virtualenv

python -m venv env

source env/bin/activate (For Windows: env\Scripts\activate)
```

Install requirements:

```
pip install -r requirements.txt
```

# Manual

# Project Structure

```
├── chatbot.py      # Main file to start/configure chatbot
├── graph.json      # Graph that representes conversation flow
├── input_to_class.csv      # Labeled user input
├── output_to_class.csv     # Labeled chatbot ouput
├── src     # Model package
│   ├── coffee.py       # Example for using callbacks
│   ├── graph.py        # Graph model 
│   ├── __init__.py
│   └── language_model.py   # Language model api
├── test # Test package
│   └── test_model.py
└── trainedModels       # Save folder for all fine-tuned language models
    └── model
```

First, we show how you could set up the coffee machine chatbot and how you can edit the conversation flow. After that we show how the framework can be used to create an own chatbot from scratch.

# Train model

Train model to classify user input based on the dataset in _input_to_class.csv_ with name coffeeModel:

```
python chatbot.py --train --inputDs='input_to_class.csv' --model=coffeeModel
```
This command loads the dataset and train the language model to classify a sentence entered by the user. E.g learn to classify "Please make tee" and "I want to tee" as MAKE_TEE. The fine-tuned model will be saved in the folder _trainedModels/coffeeModel. After training and test the confusion matrix is printed. 

To adjust the model hyperparameters, edit the file train.py.

# Start the chatbot

Start with chatbot with _coffeeModel_, _coffeeGraph.json_ and use responses from _output_to_class.csv_.

```
python chatbot.py --model=coffeeModel --outputDs=output_to_class.csv --graph=coffeeGraph.json
``` 

Then you can interact with the chatbot via command line.

# Graph.json

Each Key e.g GIVE_OPTIONS represents a state/node in the conversation graph e.g

```
"GIVE_OPTIONS": {
    "classToTransitions": {
      "WANT_COFFEE": "MAKE_COFFEE",
      "WANT_CHOCOLATE": "MAKE_CHOCOLATE",
      "WANT_TEE": "MAKE_TEE"
    },
    "sayOnEntry": "GIVE_OPTIONS",
    "sayOnExit": null,
    "nextState": null
},
```


If _sayOnEntry_ exists  and is not null, a randomly selected output from _output_to_class.csv_ labelled  as value e.g GIVE_OPTIONS is printed when conversion goes into this state. Same goes for sayOnExit. At least on output labelled as the value must exists in _output_to_class.csv_ e.g. "We have tea, coffee and hot chocolate", GIVE_OPTIONS.

The dictionary map the classified user input to the next state. E.g the current state is GIVE_OPTIONS, user input "Make Tee please" is classified AS WANT_TEE, then a transition to the state MAKE_TEE is executed.

```
  "ADD_SUGAR": {
    "nextState": "VERIFY_ORDER",
    "sayOnEntry": "CONFIRM"
  },
```

If _nextState_ exists and is not null _classToTransitions_ is ignored and the next transition goes to the state without user e.g., VERIFY_ORDER always follows ADD_SUGAR.

# Callbacks

If you want more control or combine the chatbot with other components or modules, you can inherit from the state class in _graph.py_ e.g.

```
  "ADD_SUGAR": {
    "nextState": "VERIFY_ORDER",
    "sayOnEntry": "CONFIRM"
  },
```
For this you have to name the inheriting class the same as in graph.json. Case and _ is ignored.

```python
class AddSugar(State):

    def on_entry(self):
        coffee_machine.add_order(SUGAR)

    def on_exit(self):
        pass
```

Additionally you have to import the subclasses when the graph is loaded in _graph.py_

```python
def load_graph(model_path: Path, class_to_output: dict, language_model) -> Graph:
    ...
        from src.coffee import MakeCoffee, MakeTee, MakeChocolate, VerifyOrder
    ...
```

# Set up own chatbot

Create graph template

```
python chatbot.py --createGraphTemplate STATE_1 STATE_2 STATE_N --graph=OWN_GRAPHNAME.json
```

You can define an own input/output dataset file by creating a csv file with this structure:


_own_inputs.csv_

```
user Input, Label
I want to buy this jeans, BUY_JEANS
...
```

_own_outputs.csv_

```
chatbot output, Label
Hi, GREET_USER
...
```

You can set up transitions, callbacks and responses likewise in our coffee-bot  example.


Then you need to train a model based on _own_inputs.csv_.

```
python chatbot.py --train --inputDs=own_inputs.csv --model=MY_MODELNAME
```

To check if your model is set up correctly, add verbose flag to receive information about classification and transition

```
python chatbot.py --model=MY_MODELNAME --outputDs=own_outputs.csv --graph=OWN_GRAPHNAME.json -v
``` 

```
You can get Coffee, Tea or hot chocolate. What can i serve to you?
INFO: Set next state: GIVE_OPTIONS
tea
INFO: "tea" classified as WANT_TEE
INFO: Set next state: MAKE_TEE
INFO: Set next state: ASK_MILK
```

# Tests

We recommend [pytest](https://pytest.org/) to run the unit tests. 

Installation:

```
pip install pytest
```

Run tests:

```
pytest -v
```


# Limitations

Every state uses the same fine-tuned model. If every state has an own fine-tuned model, results could be improved. Additionally, uncertainty for complete unexpected answers isn't implemented yet.


# Documentation

We recommend [pdoc](https://pdoc3.github.io/pdoc/) the generate the documentation files.

Installation:

```
pip install pdoc3
```

Generate documentation:

```
pdoc --html src
```

## License

[The Unlicense](https://choosealicense.com/licenses/unlicense/)


# Contact

If you have questions, issues or feedback: 

till.johanndeiter (at) web.de
