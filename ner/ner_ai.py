import random

import spacy
from spacy.training.example import Example

from .training_data_loader import TrainingDataLoader
from pathlib import Path

import logging
from rich.logging import RichHandler


class NerAI:
    def __init__(self, log):
        self.log = log
        self.TRAINING_DATA = []
        self.nlp = spacy.load("en_core_web_sm")
        self.ner = self.nlp.get_pipe("ner")

    def load_training_data(self, file):
        loader = TrainingDataLoader()
        self.TRAINING_DATA.extend(loader.load_json(file))

    def create_ents(self):
        for _, annotations in self.TRAINING_DATA:
            for ent in annotations.get("entities"):
                self.ner.add_label(ent[2])

    def train(self, num_itn):
        print("Starting Training")
        optimizer = self.nlp.create_optimizer()
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]

        with self.nlp.disable_pipes(*other_pipes):
            for itn in range(num_itn):
                random.shuffle(self.TRAINING_DATA)
                losses = {}

                for batch in spacy.util.minibatch(self.TRAINING_DATA, size=8):
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        self.nlp.update(
                            [example], drop=0.35, sgd=optimizer, losses=losses
                        )
                self.log.info("itn: " + str(itn) + " Losses:", losses)
                print("itn: " + str(itn) + " Losses:", losses)
        self.log.info("Final loss: ", losses)
        print("Final loss: ", losses)
        print("\n")

    def save_model(self, model_name, model_output_dir):
        output_dir = Path(model_output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        self.nlp.meta["name"] = model_name
        self.nlp.to_disk(output_dir)

    def load_model(self, model_output_dir):
        self.nlp = spacy.load(model_output_dir)


if __name__ == "__main__":
    # need to create a logger here for testing
    log = logging.getLogger("rich")
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET",
        format=FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    ai = NerAI(log)
    ai.load_training_data("item_training_data.json")
    ai.create_ents()
    ai.train(2)
