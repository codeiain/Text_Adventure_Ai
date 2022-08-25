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


class NerBasicMonsterTest(unittest.TestCase):
    def test_MONSTER_0(self):
        test_text = "Name"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Name")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_1(self):
        test_text = "Aarakocra"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Aarakocra")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_2(self):
        test_text = "Aboleth"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Aboleth")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_3(self):
        test_text = "Albino Dwarf Warrior"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Albino Dwarf Warrior")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_4(self):
        test_text = "Aldani Lobsterfolk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Aldani Lobsterfolk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_5(self):
        test_text = "Allip"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Allip")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_6(self):
        test_text = "Almiraj"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Almiraj")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_7(self):
        test_text = "Ambush Drake"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ambush Drake")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_8(self):
        test_text = "Angel Deva"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Angel Deva")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_9(self):
        test_text = "Angel Planetar"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Angel Planetar")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_10(self):
        test_text = "Angel Solar"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Angel Solar")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_11(self):
        test_text = "Animated Object Armor"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Animated Object Armor")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_12(self):
        test_text = "Animated Object Broom of Attack"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Animated Object Broom of Attack")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_13(self):
        test_text = "Animated Object Flying Sword"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Animated Object Flying Sword")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_14(self):
        test_text = "Animated Object Rug of Smothering"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Animated Object Rug of Smothering")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_15(self):
        test_text = "Animated Object Table"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Animated Object Table")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_16(self):
        test_text = "Ankheg"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ankheg")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_17(self):
        test_text = "Assassin Vine"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Assassin Vine")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_18(self):
        test_text = "Assorted Beast Aurochs"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Assorted Beast Aurochs")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_19(self):
        test_text = "Assorted Beast Cattle"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Assorted Beast Cattle")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_20(self):
        test_text = "Assorted Beast Dolphin"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Assorted Beast Dolphin")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_21(self):
        test_text = "Assorted Beast Swarm of Rot Grubs"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Assorted Beast Swarm of Rot Grubs")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_22(self):
        test_text = "Astral Dreadnought"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Astral Dreadnought")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_23(self):
        test_text = "Atropal"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Atropal")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_24(self):
        test_text = "Avatar of Death"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Avatar of Death")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_25(self):
        test_text = "Awakened Zurkhwood"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Awakened Zurkhwood")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_26(self):
        test_text = "Azer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Azer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_27(self):
        test_text = "Baba Lysaga's Creeping Hut"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Baba Lysaga's Creeping Hut")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_28(self):
        test_text = "Balhannoth"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Balhannoth")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_29(self):
        test_text = "Banderhobb"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Banderhobb")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_30(self):
        test_text = "Banshee"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Banshee")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_31(self):
        test_text = "Barghest"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Barghest")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_32(self):
        test_text = "Basilisk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Basilisk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_33(self):
        test_text = "Behir"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Behir")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_34(self):
        test_text = "Beholder"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Beholder")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_35(self):
        test_text = "Beholder Death Kiss"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Beholder Death Kiss")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_36(self):
        test_text = "Beholder Death Tyrant"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Beholder Death Tyrant")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_37(self):
        test_text = "Beholder Gauth"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Beholder Gauth")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_38(self):
        test_text = "Beholder Gazer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Beholder Gazer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_39(self):
        test_text = "Beholder Spectator"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Beholder Spectator")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_40(self):
        test_text = "Berbalang"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Berbalang")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_41(self):
        test_text = "Black Earth Guard"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Black Earth Guard")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_42(self):
        test_text = "Black Earth Priest"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Black Earth Priest")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_43(self):
        test_text = "Black Earth Burrowshark"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Black Earth Burrowshark")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_44(self):
        test_text = "Black Earth Sacred Stone Monk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Black Earth Sacred Stone Monk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_45(self):
        test_text = "Black Earth Stonemelder"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Black Earth Stonemelder")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_46(self):
        test_text = "Blight Needle"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Blight Needle")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_47(self):
        test_text = "Blight Tree"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Blight Tree")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_48(self):
        test_text = "Blight Twig"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Blight Twig")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_49(self):
        test_text = "Blight Vine"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Blight Vine")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_50(self):
        test_text = "Bodak"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Bodak")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_51(self):
        test_text = "Boggle"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Boggle")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_52(self):
        test_text = "Boneclaw"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Boneclaw")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_53(self):
        test_text = "Bugbear"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Bugbear")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_54(self):
        test_text = "Bugbear Chief"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Bugbear Chief")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_55(self):
        test_text = "Bulette"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Bulette")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_56(self):
        test_text = "Bullywug"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Bullywug")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_57(self):
        test_text = "Cadaver Collector"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Cadaver Collector")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_58(self):
        test_text = "Cambion"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Cambion")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_59(self):
        test_text = "Carrion Crawler"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Carrion Crawler")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_60(self):
        test_text = "Catoblepas"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Catoblepas")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_61(self):
        test_text = "Cave Fisher"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Cave Fisher")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_62(self):
        test_text = "Centaur"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Centaur")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_63(self):
        test_text = "Centaur Mummy"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Centaur Mummy")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_64(self):
        test_text = "Chimera"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Chimera")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_65(self):
        test_text = "Chitine"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Chitine")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_66(self):
        test_text = "Chitine Choldrith"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Chitine Choldrith")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_67(self):
        test_text = "Choker"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Choker")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_68(self):
        test_text = "Chuul"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Chuul")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_69(self):
        test_text = "Chwinga"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Chwinga")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_70(self):
        test_text = "Cloaker"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Cloaker")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_71(self):
        test_text = "Clockwork Bronze Scout"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Clockwork Bronze Scout")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_72(self):
        test_text = "Clockwork Iron Cobra"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Clockwork Iron Cobra")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_73(self):
        test_text = "Clockwork Oaken Bolter"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Clockwork Oaken Bolter")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_74(self):
        test_text = "Clockwork Stone Defender"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Clockwork Stone Defender")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_75(self):
        test_text = "Cockatrice"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Cockatrice")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_76(self):
        test_text = "Corpse Flower"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Corpse Flower")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_77(self):
        test_text = "Couatl"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Couatl")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_78(self):
        test_text = "Crag Cat"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Crag Cat")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_79(self):
        test_text = "Cranium Rat"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Cranium Rat")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_80(self):
        test_text = "Cranium Rat Swarm"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Cranium Rat Swarm")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_81(self):
        test_text = "Crawling Claw"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Crawling Claw")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_82(self):
        test_text = "Crushing Wave Priest"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Crushing Wave Priest")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_83(self):
        test_text = "Crushing Wave Reaver"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Crushing Wave Reaver")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_84(self):
        test_text = "Crushing Wave Dark Tide Knight"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Crushing Wave Dark Tide Knight")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_85(self):
        test_text = "Crushing Wave Fathomer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Crushing Wave Fathomer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_86(self):
        test_text = "Crushing Wave One-Eyed Shiver"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Crushing Wave One-Eyed Shiver")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_87(self):
        test_text = "Cyclops"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Cyclops")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_88(self):
        test_text = "Darkling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Darkling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_89(self):
        test_text = "Darkling Elder"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Darkling Elder")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_90(self):
        test_text = "Darkmantle"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Darkmantle")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_91(self):
        test_text = "Death Knight"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Death Knight")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_92(self):
        test_text = "Deathlock"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Deathlock")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_93(self):
        test_text = "Deathlock Mastermind"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Deathlock Mastermind")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_94(self):
        test_text = "Deathlock Wight"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Deathlock Wight")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_95(self):
        test_text = "Decapus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Decapus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_96(self):
        test_text = "Deep Scion"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Deep Scion")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_97(self):
        test_text = "Demilich"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demilich")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_98(self):
        test_text = "Demon Lord Baphomet"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Lord Baphomet")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_99(self):
        test_text = "Demon Lord Demogorgon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Lord Demogorgon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_100(self):
        test_text = "Demon Lord Fraz-Urb'Luu"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Lord Fraz-Urb'Luu")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_101(self):
        test_text = "Demon Lord Graz'Zt"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Lord Graz'Zt")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_102(self):
        test_text = "Fiend Shapechanger"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Fiend Shapechanger")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_103(self):
        test_text = "Demon Lord Juiblex"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Lord Juiblex")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_104(self):
        test_text = "Demon Lord Orcus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Lord Orcus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_105(self):
        test_text = "Demon Lord Yeenoghu"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Lord Yeenoghu")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_106(self):
        test_text = "Demon Lord Zuggtmoy"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Lord Zuggtmoy")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_107(self):
        test_text = "Demon Abyssal Wretch"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Abyssal Wretch")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_108(self):
        test_text = "Demon Alkilith"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Alkilith")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_109(self):
        test_text = "Demon Armanite"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Armanite")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_110(self):
        test_text = "Demon Babau"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Babau")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_111(self):
        test_text = "Demon Balor"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Balor")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_112(self):
        test_text = "Demon Barlgura"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Barlgura")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_113(self):
        test_text = "Demon Bulezau"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Bulezau")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_114(self):
        test_text = "Demon Chasme"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Chasme")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_115(self):
        test_text = "Demon Draegloth"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Draegloth")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_116(self):
        test_text = "Demon Dretch"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Dretch")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_117(self):
        test_text = "Demon Dybbuk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Dybbuk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_118(self):
        test_text = "Demon Glabrezu"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Glabrezu")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_119(self):
        test_text = "Demon Goristro"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Goristro")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_120(self):
        test_text = "Demon Hezrou"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Hezrou")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_121(self):
        test_text = "Demon Manes"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Manes")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_122(self):
        test_text = "Demon Marilith"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Marilith")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_123(self):
        test_text = "Demon Maurezhi"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Maurezhi")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_124(self):
        test_text = "Demon Maw"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Maw")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_125(self):
        test_text = "Demon Molydeus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Molydeus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_126(self):
        test_text = "Demon Nabassu"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Nabassu")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_127(self):
        test_text = "Demon Nalfeshnee"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Nalfeshnee")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_128(self):
        test_text = "Demon Quasit"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Quasit")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_129(self):
        test_text = "Demon Rutterkin"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Rutterkin")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_130(self):
        test_text = "Demon Shadow"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Shadow")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_131(self):
        test_text = "Demon Shoosuva"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Shoosuva")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_132(self):
        test_text = "Demon Sibriex"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Sibriex")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_133(self):
        test_text = "Demon Vrock"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Vrock")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_134(self):
        test_text = "Demon Wastrilith"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Wastrilith")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_135(self):
        test_text = "Demon Yochlol"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Demon Yochlol")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_136(self):
        test_text = "Derro"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Derro")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_137(self):
        test_text = "Derro Savant"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Derro Savant")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_138(self):
        test_text = "Devil Abishai Black"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Abishai Black")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_139(self):
        test_text = "Devil Abishai Blue"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Abishai Blue")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_140(self):
        test_text = "Devil Abishai Green"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Abishai Green")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_141(self):
        test_text = "Devil Abishai Red"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Abishai Red")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_142(self):
        test_text = "Devil Abishai White"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Abishai White")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_143(self):
        test_text = "Devil Amnizu"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Amnizu")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_144(self):
        test_text = "Devil Arch Bael"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Arch Bael")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_145(self):
        test_text = "Devil Arch Geryon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Arch Geryon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_146(self):
        test_text = "Devil Arch Hutijin"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Arch Hutijin")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_147(self):
        test_text = "Devil Arch Moloch"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Arch Moloch")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_148(self):
        test_text = "Devil Arch Titivilus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Arch Titivilus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_149(self):
        test_text = "Devil Arch Zariel"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Arch Zariel")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_150(self):
        test_text = "Devil Barbed"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Barbed")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_151(self):
        test_text = "Devil Bearded"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Bearded")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_152(self):
        test_text = "Devil Bone"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Bone")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_153(self):
        test_text = "Devil Chain"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Chain")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_154(self):
        test_text = "Devil Erinyes"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Erinyes")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_155(self):
        test_text = "Devil Hellfire Engine"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Hellfire Engine")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_156(self):
        test_text = "Devil Horned"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Horned")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_157(self):
        test_text = "Devil Ice"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Ice")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_158(self):
        test_text = "Devil Imp"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Imp")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_159(self):
        test_text = "Devil Lemure"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Lemure")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_160(self):
        test_text = "Devil Merregon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Merregon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_161(self):
        test_text = "Devil Narzugon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Narzugon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_162(self):
        test_text = "Devil Nupperibo"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Nupperibo")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_163(self):
        test_text = "Devil Orthon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Orthon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_164(self):
        test_text = "Devil Pit Fiend"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Pit Fiend")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_165(self):
        test_text = "Devil Spined"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devil Spined")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_166(self):
        test_text = "Devourer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Devourer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_167(self):
        test_text = "Dinosaur Allosaurus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dinosaur Allosaurus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_168(self):
        test_text = "Dinosaur Ankylosaurus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dinosaur Ankylosaurus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_169(self):
        test_text = "Dinosaur Brontosaurus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dinosaur Brontosaurus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_170(self):
        test_text = "Dinosaur Deinonychus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dinosaur Deinonychus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_171(self):
        test_text = "Dinosaur Dimetrodon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dinosaur Dimetrodon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_172(self):
        test_text = "Dinosaur Hadrosaurus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dinosaur Hadrosaurus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_173(self):
        test_text = "Dinosaur Plesiosaurus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dinosaur Plesiosaurus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_174(self):
        test_text = "Dinosaur Pteranodon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dinosaur Pteranodon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_175(self):
        test_text = "Dinosaur Quetzalcoatlus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dinosaur Quetzalcoatlus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_176(self):
        test_text = "Dinosaur Stegosaurus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dinosaur Stegosaurus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_177(self):
        test_text = "Dinosaur Triceratops"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dinosaur Triceratops")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_178(self):
        test_text = "Dinosaur Tyrannosaurus Rex"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dinosaur Tyrannosaurus Rex")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_179(self):
        test_text = "Dinosaur Velociraptor"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dinosaur Velociraptor")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_180(self):
        test_text = "Displacer Beast"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Displacer Beast")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_181(self):
        test_text = "Doppelganger"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Doppelganger")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_182(self):
        test_text = "Dracolich Template"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dracolich Template")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_183(self):
        test_text = "Dragon Chromatic Black Adult"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Black Adult")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_184(self):
        test_text = "Dragon Chromatic Black Ancient"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Black Ancient")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_185(self):
        test_text = "Dragon Chromatic Black Wyrmling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Black Wyrmling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_186(self):
        test_text = "Dragon Chromatic Black Young"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Black Young")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_187(self):
        test_text = "Dragon Chromatic Blue Adult"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Blue Adult")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_188(self):
        test_text = "Dragon Chromatic Blue Ancient"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Blue Ancient")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_189(self):
        test_text = "Dragon Chromatic Blue Wyrmling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Blue Wyrmling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_190(self):
        test_text = "Dragon Chromatic Blue Young"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Blue Young")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_191(self):
        test_text = "Dragon Chromatic Green Adult"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Green Adult")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_192(self):
        test_text = "Dragon Chromatic Green Ancient"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Green Ancient")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_193(self):
        test_text = "Dragon Chromatic Green Wyrmling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Green Wyrmling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_194(self):
        test_text = "Dragon Chromatic Green Young"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Green Young")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_195(self):
        test_text = "Dragon Chromatic Red Adult"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Red Adult")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_196(self):
        test_text = "Dragon Chromatic Red Ancient"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Red Ancient")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_197(self):
        test_text = "Dragon Chromatic Red Wyrmling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Red Wyrmling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_198(self):
        test_text = "Dragon Chromatic Red Young"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic Red Young")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_199(self):
        test_text = "Dragon Chromatic White Adult"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic White Adult")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_200(self):
        test_text = "Dragon Chromatic White Ancient"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic White Ancient")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_201(self):
        test_text = "Dragon Chromatic White Wyrmling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic White Wyrmling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_202(self):
        test_text = "Dragon Chromatic White Young"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Chromatic White Young")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_203(self):
        test_text = "Dragon Metallic Brass Adult"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Brass Adult")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_204(self):
        test_text = "Dragon Metallic Brass Ancient"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Brass Ancient")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_205(self):
        test_text = "Dragon Metallic Brass Wyrmling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Brass Wyrmling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_206(self):
        test_text = "Dragon Metallic Brass Young"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Brass Young")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_207(self):
        test_text = "Dragon Metallic Bronze Adult"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Bronze Adult")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_208(self):
        test_text = "Dragon Metallic Bronze Ancient"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Bronze Ancient")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_209(self):
        test_text = "Dragon Metallic Bronze Wyrmling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Bronze Wyrmling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_210(self):
        test_text = "Dragon Metallic Bronze Young"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Bronze Young")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_211(self):
        test_text = "Dragon Metallic Copper Adult"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Copper Adult")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_212(self):
        test_text = "Dragon Metallic Copper Ancient"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Copper Ancient")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_213(self):
        test_text = "Dragon Metallic Copper Wyrmling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Copper Wyrmling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_214(self):
        test_text = "Dragon Metallic Copper Young"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Copper Young")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_215(self):
        test_text = "Dragon Metallic Gold Adult"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Gold Adult")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_216(self):
        test_text = "Dragon Metallic Gold Ancient"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Gold Ancient")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_217(self):
        test_text = "Dragon Metallic Gold Wyrmling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Gold Wyrmling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_218(self):
        test_text = "Dragon Metallic Gold Young"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Gold Young")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_219(self):
        test_text = "Dragon Metallic Silver Adult"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Silver Adult")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_220(self):
        test_text = "Dragon Metallic Silver Ancient"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Silver Ancient")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_221(self):
        test_text = "Dragon Metallic Silver Wyrmling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Silver Wyrmling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_222(self):
        test_text = "Dragon Metallic Silver Young"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Metallic Silver Young")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_223(self):
        test_text = "Dragon Turtle"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragon Turtle")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_224(self):
        test_text = "Dragonclaw"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragonclaw")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_225(self):
        test_text = "Dragonfang"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragonfang")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_226(self):
        test_text = "Dragonsoul"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragonsoul")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_227(self):
        test_text = "Dragonwing"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dragonwing")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_228(self):
        test_text = "Dread Warrior"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dread Warrior")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_229(self):
        test_text = "Drider"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Drider")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_230(self):
        test_text = "Drow"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Drow")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_231(self):
        test_text = "Drow Arachnomancer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Drow Arachnomancer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_232(self):
        test_text = "Drow Elite Warrior"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Drow Elite Warrior")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_233(self):
        test_text = "Drow Favored Consort"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Drow Favored Consort")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_234(self):
        test_text = "Drow House Captain"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Drow House Captain")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_235(self):
        test_text = "Drow Inquisitor"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Drow Inquisitor")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_236(self):
        test_text = "Drow Mage"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Drow Mage")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_237(self):
        test_text = "Drow Matron Mother"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Drow Matron Mother")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_238(self):
        test_text = "Drow Priestess of Lolth"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Drow Priestess of Lolth")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_239(self):
        test_text = "Drow Shadowblade"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Drow Shadowblade")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_240(self):
        test_text = "Dryad"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dryad")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_241(self):
        test_text = "Duergar"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Duergar")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_242(self):
        test_text = "Duergar Darkhaft"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Duergar Darkhaft")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_243(self):
        test_text = "Duergar Despot"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Duergar Despot")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_244(self):
        test_text = "Duergar Hammerer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Duergar Hammerer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_245(self):
        test_text = "Duergar Kavalrachni"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Duergar Kavalrachni")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_246(self):
        test_text = "Duergar Keeper of the Flame"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Duergar Keeper of the Flame")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_247(self):
        test_text = "Duergar Mind Master"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Duergar Mind Master")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_248(self):
        test_text = "Duergar Screamer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Duergar Screamer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_249(self):
        test_text = "Duergar Soulblade"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Duergar Soulblade")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_250(self):
        test_text = "Duergar Spy"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Duergar Spy")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_251(self):
        test_text = "Duergar Stone Guard"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Duergar Stone Guard")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_252(self):
        test_text = "Duergar Warlord"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Duergar Warlord")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_253(self):
        test_text = "Duergar Xarrorn"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Duergar Xarrorn")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_254(self):
        test_text = "Eblis"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Eblis")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_255(self):
        test_text = "Eidolon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Eidolon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_256(self):
        test_text = "Eidolon Sacred Statue"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Eidolon Sacred Statue")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_257(self):
        test_text = "Eladrin Autumn"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Eladrin Autumn")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_258(self):
        test_text = "Eladrin Spring"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Eladrin Spring")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_259(self):
        test_text = "Eladrin Summer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Eladrin Summer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_260(self):
        test_text = "Eladrin Winter"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Eladrin Winter")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_261(self):
        test_text = "Elder Elemental Elder Tempest"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Elder Elemental Elder Tempest")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_262(self):
        test_text = "Elder Elemental Leviathan"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Elder Elemental Leviathan")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_263(self):
        test_text = "Elder Elemental Phoenix"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Elder Elemental Phoenix")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_264(self):
        test_text = "Elder Elemental Zaratan"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Elder Elemental Zaratan")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_265(self):
        test_text = "Elemental Myrmidon Air"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Elemental Myrmidon Air")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_266(self):
        test_text = "Elemental Myrmidon Earth"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Elemental Myrmidon Earth")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_267(self):
        test_text = "Elemental Myrmidon Fire"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Elemental Myrmidon Fire")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_268(self):
        test_text = "Elemental Myrmidon Water"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Elemental Myrmidon Water")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_269(self):
        test_text = "Elemental Air"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Elemental Air")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_270(self):
        test_text = "Elemental Earth"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Elemental Earth")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_271(self):
        test_text = "Elemental Fire"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Elemental Fire")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_272(self):
        test_text = "Elemental Water"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Elemental Water")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_273(self):
        test_text = "Empyrean"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Empyrean")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_274(self):
        test_text = "Eternal Flame Guardian"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Eternal Flame Guardian")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_275(self):
        test_text = "Eternal Flame Priest"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Eternal Flame Priest")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_276(self):
        test_text = "Eternal Flame Flamewrath"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Eternal Flame Flamewrath")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_277(self):
        test_text = "Eternal Flame Razerblast"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Eternal Flame Razerblast")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_278(self):
        test_text = "Ettercap"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ettercap")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_279(self):
        test_text = "Ettin"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ettin")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_280(self):
        test_text = "Faerie Dragon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Faerie Dragon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_281(self):
        test_text = "Firenewt Warlock of Imix"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Firenewt Warlock of Imix")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_282(self):
        test_text = "Firenewt Warrior"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Firenewt Warrior")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_283(self):
        test_text = "Flail Snail"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Flail Snail")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_284(self):
        test_text = "Flameskull"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Flameskull")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_285(self):
        test_text = "Flumph"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Flumph")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_286(self):
        test_text = "Flying Monkey"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Flying Monkey")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_287(self):
        test_text = "Fomorian"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Fomorian")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_288(self):
        test_text = "Froghemoth"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Froghemoth")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_289(self):
        test_text = "Fungus Gas Spore"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Fungus Gas Spore")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_290(self):
        test_text = "Fungus Shrieker"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Fungus Shrieker")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_291(self):
        test_text = "Fungus Violet"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Fungus Violet")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_292(self):
        test_text = "Galeb Duhr"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Galeb Duhr")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_293(self):
        test_text = "Gargoyle"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gargoyle")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_294(self):
        test_text = "Gargoyle Giant Four-Armed"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gargoyle Giant Four-Armed")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_295(self):
        test_text = "Genie Dao"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Genie Dao")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_296(self):
        test_text = "Genie Djinni"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Genie Djinni")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_297(self):
        test_text = "Genie Efreeti"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Genie Efreeti")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_298(self):
        test_text = "Genie Marid"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Genie Marid")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_299(self):
        test_text = "Geonid"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Geonid")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_300(self):
        test_text = "Ghost"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ghost")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_301(self):
        test_text = "Ghoul"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ghoul")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_302(self):
        test_text = "Ghoul Ghast"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ghoul Ghast")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_303(self):
        test_text = "Giant Crayfish"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Crayfish")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_304(self):
        test_text = "Giant Fly"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Fly")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_305(self):
        test_text = "Giant Ice Toad"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Ice Toad")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_306(self):
        test_text = "Giant Lightning Eel"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Lightning Eel")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_307(self):
        test_text = "Giant Snapping Turtle"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Snapping Turtle")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_308(self):
        test_text = "Giant Strider"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Strider")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_309(self):
        test_text = "Giant Subterranean Lizard"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Subterranean Lizard")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_310(self):
        test_text = "Giant Cloud"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Cloud")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_311(self):
        test_text = "Giant Cloud Smiling One"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Cloud Smiling One")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_312(self):
        test_text = "Giant Fire"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Fire")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_313(self):
        test_text = "Giant Fire Dreadnought"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Fire Dreadnought")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_314(self):
        test_text = "Giant Frost"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Frost")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_315(self):
        test_text = "Giant Frost Everlasting One"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Frost Everlasting One")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_316(self):
        test_text = "Giant Hill"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Hill")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_317(self):
        test_text = "Giant Hill Mouth of Grolantor"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Hill Mouth of Grolantor")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_318(self):
        test_text = "Giant Stone"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Stone")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_319(self):
        test_text = "Giant Stone Dreamwalker"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Stone Dreamwalker")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_320(self):
        test_text = "Giant Storm"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Storm")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_321(self):
        test_text = "Giant Storm Quintessent"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Storm Quintessent")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_322(self):
        test_text = "Gibbering Mouther"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gibbering Mouther")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_323(self):
        test_text = "Giff"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giff")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_324(self):
        test_text = "Girallon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Girallon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_325(self):
        test_text = "Githyanki Gish"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Githyanki Gish")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_326(self):
        test_text = "Githyanki Kith'Rak"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Githyanki Kith'Rak")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_327(self):
        test_text = "Githyanki Knight"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Githyanki Knight")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_328(self):
        test_text = "Githyanki Supreme Commander"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Githyanki Supreme Commander")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_329(self):
        test_text = "Githyanki Warrior"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Githyanki Warrior")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_330(self):
        test_text = "Githzerai Anarch"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Githzerai Anarch")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_331(self):
        test_text = "Githzerai Enlightened"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Githzerai Enlightened")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_332(self):
        test_text = "Githzerai Monk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Githzerai Monk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_333(self):
        test_text = "Githzerai Zerth"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Githzerai Zerth")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_334(self):
        test_text = "Gnoll"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gnoll")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_335(self):
        test_text = "Gnoll Fang of Yeenoghu"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gnoll Fang of Yeenoghu")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_336(self):
        test_text = "Gnoll Flesh Gnawer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gnoll Flesh Gnawer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_337(self):
        test_text = "Gnoll Hunter"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gnoll Hunter")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_338(self):
        test_text = "Gnoll Pack Lord"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gnoll Pack Lord")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_339(self):
        test_text = "Gnoll Witherling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gnoll Witherling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_340(self):
        test_text = "Gnoll Flind"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gnoll Flind")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_341(self):
        test_text = "Gnome Deep Svirfneblin"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gnome Deep Svirfneblin")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_342(self):
        test_text = "Goblin"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Goblin")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_343(self):
        test_text = "Goblin Boss"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Goblin Boss")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_344(self):
        test_text = "Golem Clay"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Golem Clay")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_345(self):
        test_text = "Golem Flesh"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Golem Flesh")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_346(self):
        test_text = "Golem Iron"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Golem Iron")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_347(self):
        test_text = "Golem Stone"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Golem Stone")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_348(self):
        test_text = "Gorgon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gorgon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_349(self):
        test_text = "Gray Render"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gray Render")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_350(self):
        test_text = "Grell"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Grell")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_351(self):
        test_text = "Grick"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Grick")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_352(self):
        test_text = "Grick Alpha"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Grick Alpha")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_353(self):
        test_text = "Griffon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Griffon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_354(self):
        test_text = "Grimlock"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Grimlock")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_355(self):
        test_text = "Grung"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Grung")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_356(self):
        test_text = "Grung Elite Warrior"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Grung Elite Warrior")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_357(self):
        test_text = "Grung Wildling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Grung Wildling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_358(self):
        test_text = "Guard Drake"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Guard Drake")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_359(self):
        test_text = "Guardian Portrait"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Guardian Portrait")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_360(self):
        test_text = "Hag Annis"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hag Annis")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_361(self):
        test_text = "Hag Bheur"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hag Bheur")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_362(self):
        test_text = "Hag Green"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hag Green")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_363(self):
        test_text = "Hag Night"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hag Night")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_364(self):
        test_text = "Hag Sea"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hag Sea")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_365(self):
        test_text = "Half-Dragon Template"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Half-Dragon Template")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_366(self):
        test_text = "Harpy"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Harpy")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_367(self):
        test_text = "Hell Hound"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hell Hound")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_368(self):
        test_text = "Helmed Horror"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Helmed Horror")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_369(self):
        test_text = "Hippogriff"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hippogriff")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_370(self):
        test_text = "Hobgoblin"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hobgoblin")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_371(self):
        test_text = "Hobgoblin Captain"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hobgoblin Captain")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_372(self):
        test_text = "Hobgoblin Devastator"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hobgoblin Devastator")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_373(self):
        test_text = "Hobgoblin Iron Shadow"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hobgoblin Iron Shadow")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_374(self):
        test_text = "Hobgoblin Warlord"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hobgoblin Warlord")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_375(self):
        test_text = "Homunculus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Homunculus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_376(self):
        test_text = "Hook Horror"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hook Horror")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_377(self):
        test_text = "Howler"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Howler")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_378(self):
        test_text = "Howling Hatred Initiate"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Howling Hatred Initiate")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_379(self):
        test_text = "Howling Hatred Priest"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Howling Hatred Priest")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_380(self):
        test_text = "Howling Hatred Feathergale Knight"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Howling Hatred Feathergale Knight")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_381(self):
        test_text = "Howling Hatred Hurricane"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Howling Hatred Hurricane")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_382(self):
        test_text = "Howling Hatred Skyweaver"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Howling Hatred Skyweaver")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_383(self):
        test_text = "Hulking Crab"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hulking Crab")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_384(self):
        test_text = "Hydra"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hydra")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_385(self):
        test_text = "Ice Toad"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ice Toad")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_386(self):
        test_text = "Inevitable Marut"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Inevitable Marut")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_387(self):
        test_text = "Intellect Devourer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Intellect Devourer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_388(self):
        test_text = "Invisible Stalker"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Invisible Stalker")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_389(self):
        test_text = "Ixitxachitl"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ixitxachitl")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_390(self):
        test_text = "Ixitxachitl Vampiric"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ixitxachitl Vampiric")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_391(self):
        test_text = "Jackalwere"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Jackalwere")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_392(self):
        test_text = "Jaculi"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Jaculi")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_393(self):
        test_text = "Kalka-Kylla"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kalka-Kylla")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_394(self):
        test_text = "Kamadan"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kamadan")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_395(self):
        test_text = "Kelpie"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kelpie")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_396(self):
        test_text = "Kenku"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kenku")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_397(self):
        test_text = "Ki-Rin"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ki-Rin")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_398(self):
        test_text = "Kobold"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kobold")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_399(self):
        test_text = "Kobold Dragonshield"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kobold Dragonshield")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_400(self):
        test_text = "Kobold Inventor"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kobold Inventor")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_401(self):
        test_text = "Kobold Scale Sorcerer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kobold Scale Sorcerer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_402(self):
        test_text = "Kobold Winged"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kobold Winged")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_403(self):
        test_text = "Korred"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Korred")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_404(self):
        test_text = "Kraken"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kraken")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_405(self):
        test_text = "Kraken Malformed"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kraken Malformed")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_406(self):
        test_text = "Kruthik Hive Lord"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kruthik Hive Lord")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_407(self):
        test_text = "Kruthik Adult"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kruthik Adult")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_408(self):
        test_text = "Kruthik Young"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kruthik Young")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_409(self):
        test_text = "Kuo-Toa"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kuo-Toa")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_410(self):
        test_text = "Kuo-Toa Archpriest"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kuo-Toa Archpriest")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_411(self):
        test_text = "Kuo-Toa Whip"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kuo-Toa Whip")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_412(self):
        test_text = "Lamia"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Lamia")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_413(self):
        test_text = "Leucrotta"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Leucrotta")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_414(self):
        test_text = "Lich"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Lich")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_415(self):
        test_text = "Lizard King/Queen"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Lizard King/Queen")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_416(self):
        test_text = "Lizardfolk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Lizardfolk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_417(self):
        test_text = "Lizardfolk Shaman"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Lizardfolk Shaman")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_418(self):
        test_text = "Lycanthrope Werebear"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Lycanthrope Werebear")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_419(self):
        test_text = "Lycanthrope Wereboar"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Lycanthrope Wereboar")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_420(self):
        test_text = "Lycanthrope Wererat"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Lycanthrope Wererat")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_421(self):
        test_text = "Lycanthrope Wereraven"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Lycanthrope Wereraven")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_422(self):
        test_text = "Humanoid Shapechanger"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Humanoid Shapechanger")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_423(self):
        test_text = "Lycanthrope Weretiger"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Lycanthrope Weretiger")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_424(self):
        test_text = "Lycanthrope Werewolf"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Lycanthrope Werewolf")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_425(self):
        test_text = "Magmin"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Magmin")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_426(self):
        test_text = "Manticore"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Manticore")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_427(self):
        test_text = "Mantrap"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mantrap")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_428(self):
        test_text = "Meazel"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Meazel")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_429(self):
        test_text = "Medusa"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Medusa")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_430(self):
        test_text = "Meenlock"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Meenlock")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_431(self):
        test_text = "Mephit Dust"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mephit Dust")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_432(self):
        test_text = "Mephit Ice"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mephit Ice")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_433(self):
        test_text = "Mephit Magma"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mephit Magma")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_434(self):
        test_text = "Mephit Mud"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mephit Mud")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_435(self):
        test_text = "Mephit Smoke"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mephit Smoke")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_436(self):
        test_text = "Mephit Steam"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mephit Steam")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_437(self):
        test_text = "Merfolk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Merfolk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_438(self):
        test_text = "Merrow"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Merrow")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_439(self):
        test_text = "Mimic"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mimic")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_440(self):
        test_text = "Mind Flayer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mind Flayer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_441(self):
        test_text = "Mind Flayer Lich Illithilich"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mind Flayer Lich Illithilich")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_442(self):
        test_text = "Mind Flayer Alhoon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mind Flayer Alhoon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_443(self):
        test_text = "Mind Flayer Elder Brain"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mind Flayer Elder Brain")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_444(self):
        test_text = "Mind Flayer Ulitharid"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mind Flayer Ulitharid")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_445(self):
        test_text = "Mindwitness"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mindwitness")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_446(self):
        test_text = "Minotaur"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Minotaur")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_447(self):
        test_text = "Ape"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ape")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_448(self):
        test_text = "Awakened Shrub"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Awakened Shrub")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_449(self):
        test_text = "Awakened Tree"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Awakened Tree")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_450(self):
        test_text = "Axe Beak"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Axe Beak")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_451(self):
        test_text = "Baboon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Baboon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_452(self):
        test_text = "Badger"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Badger")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_453(self):
        test_text = "Bat"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Bat")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_454(self):
        test_text = "Black Bear"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Black Bear")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_455(self):
        test_text = "Blink Dog"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Blink Dog")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_456(self):
        test_text = "Blood Hawk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Blood Hawk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_457(self):
        test_text = "Boar"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Boar")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_458(self):
        test_text = "Brown Bear"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Brown Bear")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_459(self):
        test_text = "Camel"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Camel")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_460(self):
        test_text = "Cat"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Cat")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_461(self):
        test_text = "Constrictor Snake"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Constrictor Snake")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_462(self):
        test_text = "Crab"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Crab")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_463(self):
        test_text = "Crocodile"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Crocodile")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_464(self):
        test_text = "Death Dog"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Death Dog")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_465(self):
        test_text = "Deer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Deer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_466(self):
        test_text = "Dire Wolf"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Dire Wolf")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_467(self):
        test_text = "Draft Horse"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Draft Horse")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_468(self):
        test_text = "Eagle"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Eagle")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_469(self):
        test_text = "Elephant"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Elephant")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_470(self):
        test_text = "Elk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Elk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_471(self):
        test_text = "Flying Snake"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Flying Snake")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_472(self):
        test_text = "Frog"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Frog")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_473(self):
        test_text = "Giant Ape"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Ape")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_474(self):
        test_text = "Giant Badger"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Badger")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_475(self):
        test_text = "Giant Bat"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Bat")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_476(self):
        test_text = "Giant Boar"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Boar")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_477(self):
        test_text = "Giant Centipede"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Centipede")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_478(self):
        test_text = "Giant Constrictor Snake"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Constrictor Snake")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_479(self):
        test_text = "Giant Crab"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Crab")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_480(self):
        test_text = "Giant Crocodile"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Crocodile")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_481(self):
        test_text = "Giant Eagle"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Eagle")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_482(self):
        test_text = "Giant Elk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Elk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_483(self):
        test_text = "Giant Fire Beetle"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Fire Beetle")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_484(self):
        test_text = "Giant Frog"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Frog")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_485(self):
        test_text = "Giant Goat"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Goat")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_486(self):
        test_text = "Giant Hyena"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Hyena")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_487(self):
        test_text = "Giant Lizard"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Lizard")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_488(self):
        test_text = "Giant Octopus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Octopus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_489(self):
        test_text = "Giant Owl"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Owl")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_490(self):
        test_text = "Giant Poisonous Snake"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Poisonous Snake")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_491(self):
        test_text = "Giant Rat"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Rat")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_492(self):
        test_text = "Giant Scorpion"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Scorpion")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_493(self):
        test_text = "Giant Sea Horse"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Sea Horse")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_494(self):
        test_text = "Giant Shark"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Shark")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_495(self):
        test_text = "Giant Spider"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Spider")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_496(self):
        test_text = "Giant Toad"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Toad")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_497(self):
        test_text = "Giant Vulture"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Vulture")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_498(self):
        test_text = "Giant Wasp"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Wasp")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_499(self):
        test_text = "Giant Weasel"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Weasel")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_500(self):
        test_text = "Giant Wolf Spider"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Giant Wolf Spider")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_501(self):
        test_text = "Goat"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Goat")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_502(self):
        test_text = "Hawk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hawk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_503(self):
        test_text = "Hunter Shark"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hunter Shark")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_504(self):
        test_text = "Hyena"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Hyena")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_505(self):
        test_text = "Jackal"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Jackal")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_506(self):
        test_text = "Killer Whale"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Killer Whale")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_507(self):
        test_text = "Lion"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Lion")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_508(self):
        test_text = "Lizard"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Lizard")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_509(self):
        test_text = "Mammoth"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mammoth")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_510(self):
        test_text = "Mastiff"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mastiff")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_511(self):
        test_text = "Mule"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mule")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_512(self):
        test_text = "Octopus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Octopus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_513(self):
        test_text = "Owl"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Owl")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_514(self):
        test_text = "Panther"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Panther")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_515(self):
        test_text = "Phase Spider"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Phase Spider")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_516(self):
        test_text = "Poisonous Snake"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Poisonous Snake")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_517(self):
        test_text = "Polar Bear"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Polar Bear")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_518(self):
        test_text = "Pony"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Pony")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_519(self):
        test_text = "Quipper"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Quipper")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_520(self):
        test_text = "Rat"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Rat")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_521(self):
        test_text = "Raven"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Raven")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_522(self):
        test_text = "Reef Shark"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Reef Shark")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_523(self):
        test_text = "Rhinoceros"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Rhinoceros")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_524(self):
        test_text = "Riding Horse"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Riding Horse")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_525(self):
        test_text = "Saber-Toothed Tiger"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Saber-Toothed Tiger")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_526(self):
        test_text = "Scorpion"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Scorpion")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_527(self):
        test_text = "Sea Horse"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sea Horse")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_528(self):
        test_text = "Spider"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Spider")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_529(self):
        test_text = "Swarm of Bats"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Swarm of Bats")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_530(self):
        test_text = "Swarm of Insects"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Swarm of Insects")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_531(self):
        test_text = "Swarm of Poisonous Snakes"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Swarm of Poisonous Snakes")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_532(self):
        test_text = "Swarm of Quippers"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Swarm of Quippers")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_533(self):
        test_text = "Swarm of Rats"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Swarm of Rats")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_534(self):
        test_text = "Swarm of Ravens"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Swarm of Ravens")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_535(self):
        test_text = "Tiger"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Tiger")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_536(self):
        test_text = "Vulture"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Vulture")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_537(self):
        test_text = "Warhorse"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Warhorse")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_538(self):
        test_text = "Weasel"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Weasel")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_539(self):
        test_text = "Winter Wolf"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Winter Wolf")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_540(self):
        test_text = "Wolf"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Wolf")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_541(self):
        test_text = "Worg"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Worg")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_542(self):
        test_text = "Modron Duodrone"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Modron Duodrone")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_543(self):
        test_text = "Modron Monodrone"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Modron Monodrone")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_544(self):
        test_text = "Modron Pentadrone"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Modron Pentadrone")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_545(self):
        test_text = "Modron Quadrone"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Modron Quadrone")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_546(self):
        test_text = "Modron Tridrone"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Modron Tridrone")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_547(self):
        test_text = "Mongrelfolk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mongrelfolk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_548(self):
        test_text = "Morkoth"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Morkoth")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_549(self):
        test_text = "Mummy"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mummy")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_550(self):
        test_text = "Mummy Lord"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mummy Lord")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_551(self):
        test_text = "Myconid Adult"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Myconid Adult")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_552(self):
        test_text = "Myconid Sovereign"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Myconid Sovereign")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_553(self):
        test_text = "Myconid Sprout"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Myconid Sprout")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_554(self):
        test_text = "Naergoth Bladelord"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Naergoth Bladelord")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_555(self):
        test_text = "Naga Bone"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Naga Bone")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_556(self):
        test_text = "Naga Guardian"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Naga Guardian")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_557(self):
        test_text = "Naga Spirit"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Naga Spirit")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_558(self):
        test_text = "Nagpa"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Nagpa")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_559(self):
        test_text = "Neogi"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Neogi")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_560(self):
        test_text = "Neogi Hatchling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Neogi Hatchling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_561(self):
        test_text = "Neogi Master"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Neogi Master")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_562(self):
        test_text = "Neothelid"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Neothelid")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_563(self):
        test_text = "Nereid"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Nereid")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_564(self):
        test_text = "Nightmare"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Nightmare")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_565(self):
        test_text = "Nightwalker"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Nightwalker")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_566(self):
        test_text = "Nilbog"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Nilbog")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_567(self):
        test_text = "Nothic"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Nothic")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_568(self):
        test_text = "Abjurer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Abjurer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_569(self):
        test_text = "Acolyte"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Acolyte")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_570(self):
        test_text = "Apprentice Wizard"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Apprentice Wizard")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_571(self):
        test_text = "Archdruid"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Archdruid")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_572(self):
        test_text = "Archer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Archer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_573(self):
        test_text = "Archmage"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Archmage")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_574(self):
        test_text = "Assassin"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Assassin")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_575(self):
        test_text = "Bandit"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Bandit")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_576(self):
        test_text = "Bandit Captain"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Bandit Captain")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_577(self):
        test_text = "Bard"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Bard")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_578(self):
        test_text = "Berserker"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Berserker")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_579(self):
        test_text = "Blackguard"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Blackguard")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_580(self):
        test_text = "Champion"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Champion")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_581(self):
        test_text = "Commoner"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Commoner")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_582(self):
        test_text = "Conjurer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Conjurer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_583(self):
        test_text = "Cult Fanatic"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Cult Fanatic")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_584(self):
        test_text = "Cultist"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Cultist")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_585(self):
        test_text = "Diviner"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Diviner")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_586(self):
        test_text = "Druid"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Druid")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_587(self):
        test_text = "Enchanter"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Enchanter")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_588(self):
        test_text = "Evoker"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Evoker")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_589(self):
        test_text = "Gladiator"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gladiator")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_590(self):
        test_text = "Guard"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Guard")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_591(self):
        test_text = "Illusionist"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Illusionist")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_592(self):
        test_text = "Knight"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Knight")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_593(self):
        test_text = "Kraken Priest"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Kraken Priest")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_594(self):
        test_text = "Mage"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Mage")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_595(self):
        test_text = "Martial Arts Adept"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Martial Arts Adept")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_596(self):
        test_text = "Master Thief"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Master Thief")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_597(self):
        test_text = "Necromancer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Necromancer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_598(self):
        test_text = "Noble"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Noble")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_599(self):
        test_text = "Priest"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Priest")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_600(self):
        test_text = "Scout"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Scout")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_601(self):
        test_text = "Spy"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Spy")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_602(self):
        test_text = "Swashbuckler"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Swashbuckler")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_603(self):
        test_text = "Thug"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Thug")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_604(self):
        test_text = "Transmuter"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Transmuter")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_605(self):
        test_text = "Tribal Warrior"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Tribal Warrior")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_606(self):
        test_text = "Veteran"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Veteran")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_607(self):
        test_text = "War Priest"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "War Priest")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_608(self):
        test_text = "Warlock of the Archfey"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Warlock of the Archfey")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_609(self):
        test_text = "Warlock of the Fiend"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Warlock of the Fiend")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_610(self):
        test_text = "Warlock of the Great Old One"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Warlock of the Great Old One")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_611(self):
        test_text = "Warlord"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Warlord")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_612(self):
        test_text = "Oblex Spawn"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Oblex Spawn")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_613(self):
        test_text = "Oblex Adult"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Oblex Adult")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_614(self):
        test_text = "Oblex Elder"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Oblex Elder")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_615(self):
        test_text = "Ogre"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ogre")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_616(self):
        test_text = "Ogre Battering Ram"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ogre Battering Ram")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_617(self):
        test_text = "Ogre Bolt Launcher"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ogre Bolt Launcher")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_618(self):
        test_text = "Ogre Chain Brute"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ogre Chain Brute")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_619(self):
        test_text = "Ogre Howdah"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ogre Howdah")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_620(self):
        test_text = "Ogre Half"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ogre Half")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_621(self):
        test_text = "Oni"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Oni")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_622(self):
        test_text = "Ooze Master"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ooze Master")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_623(self):
        test_text = "Black Pudding"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Black Pudding")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_624(self):
        test_text = "Gelatinous Cube"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gelatinous Cube")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_625(self):
        test_text = "Gray"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Gray")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_626(self):
        test_text = "Ochre Jelly"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ochre Jelly")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_627(self):
        test_text = "Orc"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Orc")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_628(self):
        test_text = "Orc Blade of Ilneval"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Orc Blade of Ilneval")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_629(self):
        test_text = "Orc Claw of Luthic"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Orc Claw of Luthic")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_630(self):
        test_text = "Orc Eye of Gruumsh"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Orc Eye of Gruumsh")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_631(self):
        test_text = "Orc Hand of Yurtrus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Orc Hand of Yurtrus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_632(self):
        test_text = "Orc Nurtured One of Yurtrus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Orc Nurtured One of Yurtrus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_633(self):
        test_text = "Orc Red Fang of Shargaas"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Orc Red Fang of Shargaas")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_634(self):
        test_text = "Orc War Chief"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Orc War Chief")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_635(self):
        test_text = "Orc Orog"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Orc Orog")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_636(self):
        test_text = "Orc Tanarukk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Orc Tanarukk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_637(self):
        test_text = "Otyugh"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Otyugh")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_638(self):
        test_text = "Owlbear"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Owlbear")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_639(self):
        test_text = "Pegasus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Pegasus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_640(self):
        test_text = "Peryton"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Peryton")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_641(self):
        test_text = "Phantom Warrior"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Phantom Warrior")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_642(self):
        test_text = "Piercer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Piercer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_643(self):
        test_text = "Pixie"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Pixie")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_644(self):
        test_text = "Imix"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Imix")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_645(self):
        test_text = "Ogremoch"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Ogremoch")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_646(self):
        test_text = "Olhydra"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Olhydra")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_647(self):
        test_text = "Yan-C-BIn"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yan-C-BIn")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_648(self):
        test_text = "Pseudodragon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Pseudodragon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_649(self):
        test_text = "Pterafolk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Pterafolk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_650(self):
        test_text = "Purple Worm"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Purple Worm")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_651(self):
        test_text = "Purple Worm Wormling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Purple Worm Wormling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_652(self):
        test_text = "Quaggoth"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Quaggoth")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_653(self):
        test_text = "Quickling"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Quickling")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_654(self):
        test_text = "Rakshasa"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Rakshasa")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_655(self):
        test_text = "Redcap"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Redcap")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_656(self):
        test_text = "Remorhaz"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Remorhaz")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_657(self):
        test_text = "Remorhaz Young"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Remorhaz Young")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_658(self):
        test_text = "Retriever"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Retriever")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_659(self):
        test_text = "Revenant"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Revenant")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_660(self):
        test_text = "Roc"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Roc")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_661(self):
        test_text = "Roper"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Roper")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_662(self):
        test_text = "Rust Monster"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Rust Monster")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_663(self):
        test_text = "Sahuagin"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sahuagin")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_664(self):
        test_text = "Sahuagin Baron"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sahuagin Baron")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_665(self):
        test_text = "Sahuagin Priestess"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sahuagin Priestess")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_666(self):
        test_text = "Salamander"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Salamander")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_667(self):
        test_text = "Salamander Fire Snake"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Salamander Fire Snake")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_668(self):
        test_text = "Salamander Frost"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Salamander Frost")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_669(self):
        test_text = "Satyr"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Satyr")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_670(self):
        test_text = "Scarecrow"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Scarecrow")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_671(self):
        test_text = "Sea Lion"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sea Lion")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_672(self):
        test_text = "Sea Spawn"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sea Spawn")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_673(self):
        test_text = "Shadar-Kai Gloom Weaver"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Shadar-Kai Gloom Weaver")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_674(self):
        test_text = "Shadar-Kai Shadow Dancer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Shadar-Kai Shadow Dancer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_675(self):
        test_text = "Shadar-Kai Soul Monger"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Shadar-Kai Soul Monger")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_676(self):
        test_text = "Shadow"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Shadow")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_677(self):
        test_text = "Shadow Dragon Template"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Shadow Dragon Template")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_678(self):
        test_text = "Shadow Mastiff"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Shadow Mastiff")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_679(self):
        test_text = "Shadow Mastiff Alpha"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Shadow Mastiff Alpha")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_680(self):
        test_text = "Shambling Mound"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Shambling Mound")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_681(self):
        test_text = "Shield Guardian"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Shield Guardian")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_682(self):
        test_text = "Siren"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Siren")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_683(self):
        test_text = "Skeleton"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Skeleton")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_684(self):
        test_text = "Skeleton Giant"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Skeleton Giant")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_685(self):
        test_text = "Skeleton Minotaur"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Skeleton Minotaur")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_686(self):
        test_text = "Skeleton Warhorse"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Skeleton Warhorse")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_687(self):
        test_text = "Skulk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Skulk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_688(self):
        test_text = "Skull Lord"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Skull Lord")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_689(self):
        test_text = "Slaad Tadpole"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Slaad Tadpole")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_690(self):
        test_text = "Slaad Blue"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Slaad Blue")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_691(self):
        test_text = "Slaad Death"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Slaad Death")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_692(self):
        test_text = "Slaad Gray"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Slaad Gray")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_693(self):
        test_text = "Slaad Green"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Slaad Green")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_694(self):
        test_text = "Slaad Red"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Slaad Red")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_695(self):
        test_text = "Slithering Tracker"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Slithering Tracker")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_696(self):
        test_text = "Sorrowsworn The Angry"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sorrowsworn The Angry")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_697(self):
        test_text = "Sorrowsworn The Hungry"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sorrowsworn The Hungry")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_698(self):
        test_text = "Sorrowsworn The Lonely"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sorrowsworn The Lonely")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_699(self):
        test_text = "Sorrowsworn The Lost"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sorrowsworn The Lost")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_700(self):
        test_text = "Sorrowsworn The Wretched"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sorrowsworn The Wretched")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_701(self):
        test_text = "Spawn of Kyuss"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Spawn of Kyuss")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_702(self):
        test_text = "Specter"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Specter")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_703(self):
        test_text = "Sphinx Andro"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sphinx Andro")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_704(self):
        test_text = "Sphinx Gyno"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sphinx Gyno")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_705(self):
        test_text = "Spore Servant Template"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Spore Servant Template")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_706(self):
        test_text = "Sprite"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sprite")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_707(self):
        test_text = "Star Spawn Grue"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Star Spawn Grue")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_708(self):
        test_text = "Star Spawn Hulk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Star Spawn Hulk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_709(self):
        test_text = "Star Spawn Larva Mage"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Star Spawn Larva Mage")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_710(self):
        test_text = "Star Spawn Mangler"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Star Spawn Mangler")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_711(self):
        test_text = "Star Spawn Seer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Star Spawn Seer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_712(self):
        test_text = "Steeder Female"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Steeder Female")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_713(self):
        test_text = "Steeder Male"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Steeder Male")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_714(self):
        test_text = "Steel Predator"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Steel Predator")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_715(self):
        test_text = "Stirge"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Stirge")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_716(self):
        test_text = "Stone Cursed"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Stone Cursed")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_717(self):
        test_text = "Stone Juggernaut"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Stone Juggernaut")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_718(self):
        test_text = "Strahd's Animated Armor"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Strahd's Animated Armor")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_719(self):
        test_text = "Su-Monster"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Su-Monster")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_720(self):
        test_text = "Succubus/Incubus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Succubus/Incubus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_721(self):
        test_text = "Sword Wraith Commander"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sword Wraith Commander")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_722(self):
        test_text = "Sword Wraith Warrior"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Sword Wraith Warrior")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_723(self):
        test_text = "Tabaxi Hunter"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Tabaxi Hunter")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_724(self):
        test_text = "Tabaxi Minstrel"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Tabaxi Minstrel")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_725(self):
        test_text = "Tarrasque"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Tarrasque")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_726(self):
        test_text = "Tecuziztecatl"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Tecuziztecatl")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_727(self):
        test_text = "Thayan Apprentice"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Thayan Apprentice")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_728(self):
        test_text = "Thayan Warrior"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Thayan Warrior")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_729(self):
        test_text = "Thorn Slinger"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Thorn Slinger")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_730(self):
        test_text = "Thorny"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Thorny")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_731(self):
        test_text = "Thri-Kreen"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Thri-Kreen")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_732(self):
        test_text = "Tiamat"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Tiamat")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_733(self):
        test_text = "Tiny Servant"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Tiny Servant")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_734(self):
        test_text = "Tlincalli"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Tlincalli")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_735(self):
        test_text = "Topi"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Topi")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_736(self):
        test_text = "Tortle"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Tortle")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_737(self):
        test_text = "Tortle Druid"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Tortle Druid")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_738(self):
        test_text = "Trapper"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Trapper")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_739(self):
        test_text = "Treant"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Treant")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_740(self):
        test_text = "Tressym"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Tressym")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_741(self):
        test_text = "Tri-Flower Frond"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Tri-Flower Frond")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_742(self):
        test_text = "Troglodyte"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Troglodyte")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_743(self):
        test_text = "Troglodyte Champion of Laogzed"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Troglodyte Champion of Laogzed")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_744(self):
        test_text = "Troll"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Troll")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_745(self):
        test_text = "Troll Dire"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Troll Dire")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_746(self):
        test_text = "Troll Rot"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Troll Rot")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_747(self):
        test_text = "Troll Spirit"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Troll Spirit")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_748(self):
        test_text = "Troll Venom"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Troll Venom")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_749(self):
        test_text = "Umber Hulk"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Umber Hulk")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_750(self):
        test_text = "Unicorn"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Unicorn")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_751(self):
        test_text = "Uthgardt Shaman"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Uthgardt Shaman")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_752(self):
        test_text = "Vampire"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Vampire")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_753(self):
        test_text = "Vampire Spawn"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Vampire Spawn")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_754(self):
        test_text = "Vampiric Mist"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Vampiric Mist")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_755(self):
        test_text = "Vargouille"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Vargouille")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_756(self):
        test_text = "Vegepygmy"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Vegepygmy")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_757(self):
        test_text = "Vegepygmy Chief"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Vegepygmy Chief")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_758(self):
        test_text = "Water Weird"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Water Weird")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_759(self):
        test_text = "White Maw"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "White Maw")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_760(self):
        test_text = "Wight"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Wight")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_761(self):
        test_text = "Will-O'-Wisp"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Will-O'-Wisp")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_762(self):
        test_text = "Wood Woad"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Wood Woad")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_763(self):
        test_text = "Wraith"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Wraith")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_764(self):
        test_text = "Wyvern"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Wyvern")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_765(self):
        test_text = "Xorn"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Xorn")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_766(self):
        test_text = "Xvart"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Xvart")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_767(self):
        test_text = "Xvart Speaker"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Xvart Speaker")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_768(self):
        test_text = "Xvart Warlock of Raxivort"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Xvart Warlock of Raxivort")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_769(self):
        test_text = "Yakfolk Priest"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yakfolk Priest")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_770(self):
        test_text = "Yakfolk Warrior"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yakfolk Warrior")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_771(self):
        test_text = "Yellow Musk Creeper"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yellow Musk Creeper")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_772(self):
        test_text = "Yellow Musk Zombie"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yellow Musk Zombie")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_773(self):
        test_text = "Yeth Hound"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yeth Hound")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_774(self):
        test_text = "Yeti"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yeti")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_775(self):
        test_text = "Yeti Abominable"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yeti Abominable")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_776(self):
        test_text = "Yuan-Ti Abomination"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yuan-Ti Abomination")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_777(self):
        test_text = "Yuan-Ti Anathema"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yuan-Ti Anathema")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_778(self):
        test_text = "Monstrosity Yuan-Ti"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Monstrosity Yuan-Ti")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_779(self):
        test_text = "Yuan-Ti Broodguard"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yuan-Ti Broodguard")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_780(self):
        test_text = "Yuan-Ti Malison"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yuan-Ti Malison")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_781(self):
        test_text = "Yuan-Ti Mind Whisperer"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yuan-Ti Mind Whisperer")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_782(self):
        test_text = "Monstrosity Yuan-Ti"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Monstrosity Yuan-Ti")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_783(self):
        test_text = "Yuan-Ti Nightmare Speaker"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yuan-Ti Nightmare Speaker")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_784(self):
        test_text = "Monstrosity Yuan-Ti"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Monstrosity Yuan-Ti")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_785(self):
        test_text = "Yuan-Ti Pit Master"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yuan-Ti Pit Master")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_786(self):
        test_text = "Monstrosity Yuan-Ti"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Monstrosity Yuan-Ti")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_787(self):
        test_text = "Yuan-Ti Pureblood"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yuan-Ti Pureblood")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_788(self):
        test_text = "Yugoloth Arcana"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yugoloth Arcana")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_789(self):
        test_text = "Yugoloth Cano"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yugoloth Cano")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_790(self):
        test_text = "Yugoloth Dhergo"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yugoloth Dhergo")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_791(self):
        test_text = "Yugoloth Hydro"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yugoloth Hydro")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_792(self):
        test_text = "Yugoloth Merreno"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yugoloth Merreno")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_793(self):
        test_text = "Yugoloth Mezzo"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yugoloth Mezzo")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_794(self):
        test_text = "Yugoloth Nyca"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yugoloth Nyca")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_795(self):
        test_text = "Yugoloth Oino"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yugoloth Oino")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_796(self):
        test_text = "Yugoloth Ultro"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yugoloth Ultro")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_797(self):
        test_text = "Yugoloth Yagno"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Yugoloth Yagno")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_798(self):
        test_text = "Zombie"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Zombie")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_799(self):
        test_text = "Zombie Ankylosaurus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Zombie Ankylosaurus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_800(self):
        test_text = "Zombie Beholder"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Zombie Beholder")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_801(self):
        test_text = "Zombie Girallon"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Zombie Girallon")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_802(self):
        test_text = "Zombie Greater"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Zombie Greater")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_803(self):
        test_text = "Zombie Ogre"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Zombie Ogre")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_804(self):
        test_text = "Zombie Strahd"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Zombie Strahd")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_805(self):
        test_text = "Zombie Tyrannosaurus"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Zombie Tyrannosaurus")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_806(self):
        test_text = "Zorbo"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Zorbo")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_807(self):
        test_text = "Zuggtmoy Bridesmaid"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Zuggtmoy Bridesmaid")
        self.assertEqual(doc.ents[0].label_, "MONSTER")

    def test_MONSTER_808(self):
        test_text = "Zuggtmoy Chamberlain"
        doc = ai.nlp(test_text)
        self.assertEqual(str(doc.ents[0]), "Zuggtmoy Chamberlain")
        self.assertEqual(doc.ents[0].label_, "MONSTER")


if __name__ == "__main__":

    unittest.main()
