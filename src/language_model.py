import json
from pathlib import Path
from random import shuffle
from shutil import rmtree

from numpy import argmax

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
# Do not remove this import
import tensorflow_text as text

import tensorflow_hub as hub

CHECKPOINT_FILENAME = 'checkpoint'
DICTIONARIES_FILENAME = 'dictionaries'
CLASSES_TO_ID_DICT = 'classes_to_id'
ID_TO_CLASSES_DICT = 'id_to_classes'


def create_model(num_classes: int):
    text_input = Input(shape=(), dtype=tf.string)
    preprocessor = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    encoder_inputs = preprocessor(text_input)

    encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2",
        trainable=True)
    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]

    output = Dense(num_classes, activation='softmax')(pooled_output)

    model = Model(inputs=text_input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001),
                  metrics='accuracy')

    return model


class LanguageModelApi:

    def __init__(self, model_filepath=None):

        if model_filepath is None:
            self.model = None
            self.classes_to_id = None
            self.id_to_class = None
        else:
            assert isinstance(model_filepath, Path)
            assert model_filepath.is_dir()
            assert model_filepath.joinpath(CHECKPOINT_FILENAME).exists()
            assert model_filepath.joinpath(DICTIONARIES_FILENAME).exists()

            with open(model_filepath.joinpath(DICTIONARIES_FILENAME), 'r') as file:
                import_dict = json.load(file)
                assert CLASSES_TO_ID_DICT in import_dict
                assert ID_TO_CLASSES_DICT in import_dict
                self.id_to_class = import_dict[ID_TO_CLASSES_DICT]
                self.id_to_class = {int(k): v for k, v in self.id_to_class.items()}
                self.classes_to_id = import_dict[CLASSES_TO_ID_DICT]

            self.model = create_model(len(self.classes_to_id))
            self.model.load_weights(model_filepath.joinpath(CHECKPOINT_FILENAME))

    def classify_sentence(self, sentence: str, allowed_classes=None):
        self.__assert_model_trained()
        sentence = tf.constant([sentence])
        softmax_output = self.model(sentence)

        if allowed_classes is not None:
            not_allowed_classes = filter(lambda a: a not in allowed_classes,
                                         self.classes_to_id.keys())
            for cls in not_allowed_classes:
                softmax_output[self.classes_to_id[cls]] = 0

        id_of_max_class = argmax(softmax_output)
        return self.id_to_class[id_of_max_class]

    def __assert_model_trained(self):
        assert self.model is not None
        assert self.classes_to_id is not None
        assert self.id_to_class is not None

    def train_model(self, dataset):
        shuffle(dataset)
        all_classes = set(cls for _, cls in dataset)

        self.model = create_model(len(all_classes))
        self.model.summary()

        test_train_split = int(0.8 * len(dataset))

        train_dataset = dataset[:test_train_split]
        test_dataset = dataset[test_train_split:]

        self.classes_to_id = {cls: id for id, cls in enumerate(all_classes)}

        assert len(self.classes_to_id.values()) == len(set(self.classes_to_id.values()))

        self.id_to_class = {id: cls for cls, id in self.classes_to_id.items()}

        train_x = tf.constant([sample for sample, cls in train_dataset])
        train_y = to_categorical([self.classes_to_id[cls] for sample, cls in train_dataset],
                                 num_classes=len(all_classes))

        test_x = tf.constant([sample for sample, cls in test_dataset])
        test_y = to_categorical([self.classes_to_id[cls] for sample, cls in test_dataset],
                                num_classes=len(all_classes))

        self.model.fit(x=train_x, y=train_y,
                       batch_size=1, epochs=3,
                       # callbacks=[EarlyStopping()],
                       validation_split=0.2)

        scores = self.model.evaluate(x=test_x, y=test_y, verbose=1)

        print(f'Test accuracy: {scores[1]}')

        labels = [self.classes_to_id[a[1]] for a in dataset]

        predictions = self.model(tf.constant([a[0] for a in dataset]))
        predictions = tf.math.argmax(predictions, 1).numpy()

        confusion_matrix = tf.math.confusion_matrix(labels=labels, predictions=predictions,
                                                    num_classes=len(all_classes))

        print('Confusion matrix of complete dataset:')
        print(confusion_matrix.numpy())

    def save_model(self, filepath: Path):
        self.__assert_model_trained()
        if filepath.exists():
            rmtree(str(filepath))
        filepath.mkdir()
        self.model.save_weights(filepath.joinpath(CHECKPOINT_FILENAME))

        with open(filepath.joinpath(DICTIONARIES_FILENAME), 'w') as file:
            export_dict = {CLASSES_TO_ID_DICT: self.classes_to_id,
                           ID_TO_CLASSES_DICT: self.id_to_class}
            file.write(json.dumps(export_dict))
