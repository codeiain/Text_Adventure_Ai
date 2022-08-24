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
    ai.load_training_data("./item_training_data.json")
    ai.load_training_data("./weapon_training_data.json")
    ai.load_training_data("./monster_training_data.json")
    ai.create_ents()
    ai.train(120)
    ai.save_model("ner", "models")


class NerBasicItemTest(unittest.TestCase):
    def test_item_abacus(self):
        test_text = "abacus"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "abacus")  # add assertion here

    def test_item_acid(self):
        test_text = "acid"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "acid")

    def test_item_alchemists_fire(self):
        test_text = "alchemist's fire"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "alchemist's fire")

    def test_item_antitoxin(self):
        test_text = "antitoxin"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "antitoxin")

    def test_item_backpack(self):
        test_text = "backpack"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "backpack")

    def test_item_ball_bearings(self):
        test_text = "ball bearings"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "ball bearings")

    def test_item_barrel(self):
        test_text = "barrel"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "barrel")

    def test_item_basket(self):
        test_text = "basket"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "basket")

    def test_item_bedroll(self):
        test_text = "bedroll"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "bedroll")

    def test_item_bell(self):
        test_text = "bell"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "bell")

    def test_item_blanket(self):
        test_text = "blanket"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "blanket")

    def test_item_block_and_tackle(self):
        test_text = "block and tackle"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "block and tackle")

    def test_item_book(self):
        test_text = "book"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "book")

    def test_item_glass_bottle(self):
        test_text = "glass bottle"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "glass bottle")

    def test_item_bucket(self):
        test_text = "bucket"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "bucket")

    def test_item_caltrop(self):
        test_text = "caltrop"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "caltrop")

    def test_item_candle(self):
        test_text = "candle"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "candle")

    def test_item_chain(self):
        test_text = "chain"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "chain")

    def test_item_chalk(self):
        test_text = "chalk"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "chalk")

    def test_item_chest(self):
        test_text = "chest"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "chest")

    def test_item_crowbar(self):
        test_text = "crowbar"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "crowbar")

    def test_item_fishing_tackle(self):
        test_text = "fishing tackle"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "fishing tackle")

    def test_item_flask(self):
        test_text = "flask"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "flask")

    def test_item_tankard(self):
        test_text = "tankard"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "tankard")

    def test_item_grappling_hook(self):
        test_text = "grappling hook"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "grappling hook")

    def test_item_hammer(self):
        test_text = "hammer"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "hammer")

    def test_item_sledge_hammer(self):
        test_text = "sledge hammer"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "sledge hammer")

    def test_item_hourglass(self):
        test_text = "hourglass"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "hourglass")

    def test_item_hunting_trap(self):
        test_text = "hunting trap"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "hunting trap")

    def test_item_ink(self):
        test_text = "ink"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "ink")

    def test_item_jug(self):
        test_text = "jug"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "jug")

    def test_item_pitcher(self):
        test_text = "pitcher"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "pitcher")

    def test_item_ladder(self):
        test_text = "ladder"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "ladder")

    def test_item_lamp(self):
        test_text = "lamp"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "lamp")

    def test_item_bullseye_lantern(self):
        test_text = "bullseye lantern"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "bullseye lantern")

    def test_item_hooded_lantern(self):
        test_text = "hooded lantern"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "hooded lantern")

    def test_item_lock(self):
        test_text = "lock"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "lock")

    def test_item_magnifying_glass(self):
        test_text = "magnifying glass"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "magnifying glass")

    def test_item_manacles(self):
        test_text = "manacles"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "manacles")

    def test_item_mess_kit(self):
        test_text = "mess kit"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "mess kit")

    def test_item_steel_mirror(self):
        test_text = "steel mirror"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "steel mirror")

    def test_item_mirror(self):
        test_text = "mirror"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "mirror")

    def test_item_flask_of_oil(self):
        test_text = "flask of oil"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "flask of oil")

    def test_item_oil(self):
        test_text = "oil"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "oil")

    def test_item_paper(self):
        test_text = "paper"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "paper")

    def test_item_parchment(self):
        test_text = "parchment"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "parchment")

    def test_item_perfume(self):
        test_text = "perfume"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "perfume")

    def test_item_miners_pick(self):
        test_text = "miner's pick"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "miner's pick")

    def test_item_piton(self):
        test_text = "piton"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "piton")

    def test_item_iron_pot(self):
        test_text = "iron pot"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "iron pot")

    def test_item_pouch(self):
        test_text = "pouch"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "pouch")

    def test_item_quiver(self):
        test_text = "quiver"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "quiver")

    def test_item_portable_ram(self):
        test_text = "portable ram"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "portable ram")

    def test_item_sack(self):
        test_text = "sack"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "sack")

    def test_item_merchants_scale(self):
        test_text = "merchant's scale"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "merchant's scale")

    def test_item_sealing_wax(self):
        test_text = "sealing wax"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "sealing wax")

    def test_item_shovel(self):
        test_text = "shovel"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "shovel")

    def test_item_signal_whistle(self):
        test_text = "signal whistle"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "signal whistle")

    def test_item_signet_ring(self):
        test_text = "signet ring"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "signet ring")

    def test_item_soap(self):
        test_text = "soap"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "soap")

    def test_item_spellbook(self):
        test_text = "spellbook"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "spellbook")

    def test_item_iron_spikes(self):
        test_text = "iron spikes"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "iron spikes")

    def test_item_spyglass(self):
        test_text = "spyglass"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "spyglass")

    def test_item_tent(self):
        test_text = "tent"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "tent")

    def test_item_torch(self):
        test_text = "torch"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "torch")

    def test_item_waterskin(self):
        test_text = "waterskin"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "waterskin")

    def test_item_whetstone(self):
        test_text = "whetstone"
        doc = ai.nlp(test_text)
        self.assertEqual(doc.ents[0].label_, "ITEM")
        self.assertEqual(str(doc.ents[0]), "whetstone")


if __name__ == "__main__":
    unittest.main()
