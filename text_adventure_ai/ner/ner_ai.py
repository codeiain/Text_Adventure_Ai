import random

import spacy
from spacy.training.example import Example

from .training_data_loader import TrainingDataLoader
from pathlib import Path

import logging

import time
from rich.logging import RichHandler

from os import path

basedir = path.abspath(path.dirname(__file__))


class NerAI:
    def __init__(self, log, prometheus_gauge=None, prometheus_summary=None):
        self.log = log
        self.log.info("New AI")
        self.TRAINING_DATA = []
        self.nlp = spacy.load("en_core_web_lg")
        self.ner = self.nlp.get_pipe("ner")
        self.prometheus_gauge = prometheus_gauge
        self.prometheus_summary = prometheus_summary

    def load_training_data(self, file):
        loader = TrainingDataLoader()
        self.TRAINING_DATA.extend(loader.load_json(file))

    def create_ents(self):
        for _, annotations in self.TRAINING_DATA:
            for ent in annotations.get("entities"):
                self.ner.add_label(ent[2])

    def train(self, iteration, num_itn):
        self.load_training_data("./training_data/item_training_data.json")
        self.load_training_data("./training_data/weapon_training_data.json")
        self.load_training_data("./training_data/monster_training_data.json")
        self.load_training_data("./training_data/direction_training_data.json")
        self.create_ents()
        self.log.info("Starting Training")
        optimizer = self.nlp.create_optimizer()
        return_int = iteration
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "ner"]
        with self.nlp.disable_pipes(*other_pipes):
            for itn in range(iteration, (iteration + num_itn)):
                start_time = time.time()
                self.log.info("Starting " + str(itn))
                random.shuffle(self.TRAINING_DATA)
                losses = {}
                for batch in spacy.util.minibatch(self.TRAINING_DATA, size=8):
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        self.nlp.update(
                            [example], drop=0.35, sgd=optimizer, losses=losses
                        )
                if self.prometheus_gauge != None:
                    self.prometheus_gauge.labels("itn").set(itn)
                    self.prometheus_gauge.labels("loss").set(losses["ner"])

                self.log.info("itn: " + str(itn) + " Losses:" + str(losses["ner"]))
                return_int = return_int + 1
                end_time = time.time()
                self.log.info("train time: {}".format(end_time - start_time))
                self.prometheus_gauge.labels("itn_time").set(end_time - start_time)
                if self.prometheus_summary != None:
                    self.prometheus_summary.observe(end_time - start_time)
        self.log.info("Final loss: " + str(losses["ner"]))
        if self.prometheus_gauge != None:
            self.prometheus_gauge.labels("final_loss").set(losses["ner"])
        return return_int, losses["ner"]

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
    ai.create_ents()
    ai.train(2)
