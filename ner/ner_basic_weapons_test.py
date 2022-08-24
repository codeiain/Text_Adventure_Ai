import unittest
import logging
import os
import pytest
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
    ai.load_training_data("./item_training_data.json")
    ai.load_training_data("./weapon_training_data.json")
    ai.load_training_data("./monster_training_data.json")
    ai.create_ents()
    ai.train(120)
    ai.save_model("ner", "models")


class NerBasicWeponsTest(unittest.TestCase):
    def test_weapon_club(self):
        test_text = "club"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "club")

    def test_weapon_dagger(self):
        test_text = "dagger"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "dagger")

    def test_weapon_greatclub(self):
        test_text = "greatclub"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "greatclub")

    def test_weapon_handaxe(self):
        test_text = "handaxe"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "handaxe")

    def test_weapon_javelin(self):
        test_text = "javelin"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "javelin")

    def test_weapon_light_hammer(self):
        test_text = "light hammer"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "light hammer")

    def test_weapon_mace(self):
        test_text = "mace"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "mace")

    def test_weapon_quarterstaff(self):
        test_text = "quarterstaff"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "quarterstaff")

    def test_weapon_sickle(self):
        test_text = "sickle"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "sickle")

    def test_weapon_spear(self):
        test_text = "spear"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "spear")

    def test_weapon_light_crossbow(self):
        test_text = "light crossbow"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "light crossbow")

    def test_weapon_dart(self):
        test_text = "dart"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "dart")

    def test_weapon_shortbow(self):
        test_text = "shortbow"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "shortbow")

    def test_weapon_sling(self):
        test_text = "sling"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "sling")

    def test_weapon_battleaxe(self):
        test_text = "battleaxe"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "battleaxe")

    def test_weapon_flail(self):
        test_text = "flail"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "flail")

    def test_weapon_glaive(self):
        test_text = "glaive"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "glaive")

    def test_weapon_greataxe(self):
        test_text = "greataxe"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "greataxe")

    def test_weapon_greatsword(self):
        test_text = "greatsword"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "greatsword")

    def test_weapon_halberd(self):
        test_text = "halberd"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "halberd")

    def test_weapon_lance(self):
        test_text = "lance"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "lance")

    def test_weapon_longsword(self):
        test_text = "longsword"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "longsword")

    def test_weapon_maul(self):
        test_text = "maul"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "maul")

    def test_weapon_morningstar(self):
        test_text = "morningstar"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "morningstar")

    def test_weapon_pike(self):
        test_text = "pike"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "pike")

    def test_weapon_rapier(self):
        test_text = "rapier"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "rapier")

    def test_weapon_scimitar(self):
        test_text = "scimitar"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "scimitar")

    def test_weapon_shortsword(self):
        test_text = "shortsword"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "shortsword")

    def test_weapon_trident(self):
        test_text = "trident"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "trident")

    def test_weapon_war_pick(self):
        test_text = "war pick"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "war pick")

    def test_weapon_warhammer(self):
        test_text = "warhammer"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "warhammer")

    def test_weapon_whip(self):
        test_text = "whip"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "whip")

    def test_weapon_blowgun(self):
        test_text = "blowgun"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "blowgun")

    def test_weapon_hand_crossbow(self):
        test_text = "hand crossbow"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "hand crossbow")

    def test_weapon_heavy_crossbow(self):
        test_text = "heavy crossbow"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "heavy crossbow")

    def test_weapon_longbow(self):
        test_text = "longbow"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "longbow")

    def test_weapon_net(self):
        test_text = "net"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "WEAPON")
        self.assertEqual(str(doc.ents[0]), "net")


if __name__ == "__main__":
    unittest.main()
