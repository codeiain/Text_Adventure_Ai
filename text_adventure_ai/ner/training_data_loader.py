import json
import pkg_resources

class ents:
    def __init__(self, target, type_ent):
        self.target = target
        self.type_ent = type_ent


class TrainingDataLoader:
    def __int__(self):
        pass

    def load_json(self, filepath):
        stream = pkg_resources.resource_stream(__name__, filepath).read().decode()
        jsonFile = json.loads(stream)
        return self.covert(jsonFile)

    def covert(self, list):
        tr = []
        for item in list:
            entites = []
            for en in item["ents"]:
                entites.append(ents(en["target"], en["ent_type"]))
            tr.append(self.build_spacy_ner(item["name"], entites))

        return tr

    def build_spacy_ner(self, text, entities):
        all_ents = []
        for entity in entities:
            start = str.find(text, entity.target)
            end = start + len(entity.target)
            all_ents += [(start, end, entity.type_ent)]
        return (text, {"entities": all_ents})


if __name__ == "__main__":
    training = TrainingDataLoader()
    TRAINING_DATA = training.load_json("training_data\item_training_data.json")
