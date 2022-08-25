import unittest
import logging
import os

from rich.logging import RichHandler
from .ner_ai import NerAI

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
if os.path.isdir("models"):
    ai.load_model("models")
else:
    ai.load_training_data("./training_data/item_training_data.json")
    ai.load_training_data("./training_data/weapon_training_data.json")
    ai.load_training_data("./training_data/monster_training_data.json")
    ai.load_training_data("./training_data/direction_training_data.json")
    ai.create_ents()
    ai.train(10)
    ai.save_model("ner", "models")


class NerBasicDirectionTest(unittest.TestCase):
    def test_direction_north(self):
        test_text = "north"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "DIRECTION")
        self.assertEqual(str(doc.ents[0]), "north")  # add assertion

    def test_direction_south(self):
        test_text = "south"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "DIRECTION")
        self.assertEqual(str(doc.ents[0]), "south")  # add assertion

    def test_direction_east(self):
        test_text = "east"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "DIRECTION")
        self.assertEqual(str(doc.ents[0]), "east")  # add assertion

    def test_direction_west(self):
        test_text = "west"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "DIRECTION")
        self.assertEqual(str(doc.ents[0]), "west")  # add assertion

    def test_direction_up(self):
        test_text = "up"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "DIRECTION")
        self.assertEqual(str(doc.ents[0]), "up")  # add assertion

    def test_direction_down(self):
        test_text = "down"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "DIRECTION")
        self.assertEqual(str(doc.ents[0]), "down")  # add assertion

    def test_direction_left(self):
        test_text = "left"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "DIRECTION")
        self.assertEqual(str(doc.ents[0]), "left")  # add assertion

    def test_direction_right(self):
        test_text = "right"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "DIRECTION")
        self.assertEqual(str(doc.ents[0]), "right")  # add assertion
