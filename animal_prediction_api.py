from fastapi import FastAPI, UploadFile, File
from animal_classification import AnimalClassificationModel

app = FastAPI()

animal_dict = {}
animal_types = []
static_pet_types = ["Cattle", "Goat", "Sheep", "Dog", "Possum", "Donkey", "Horse", "Cat", "Rabbit", "Guinea Pig", "Pig"]
topLevelModel = AnimalClassificationModel("TopLevelClassifier", "TopLevelClassifierData", False)
perOrderModels:dict[str,AnimalClassificationModel] = {}
for order in topLevelModel.classes:
     perOrderModels[order] = AnimalClassificationModel(f"{order.title()}Classifier", f"PerOrderClassifierData/{order}", False)
with open("./animal_types.txt","r") as f:
        lines = f.read().split("\n")
        lines.remove('')
        for line in lines:
            data = line.split(": ")
            animals = data[1]
            if "," in data[1]:
                animals = data[1].split(", ")
            else:
                animals = [data[1]]
            for animal in animals:
                 animal_types.append(animal.title())
            animal_dict[data[0]] = animals

@app.post("/identifyAnimal")
def identify_animal(image_file: UploadFile = File(...)):
    file = image_file.file.read()
    animal_order = topLevelModel.predict(file)
    if animal_order != None:
        animals_in_order = animal_dict[animal_order] 
        if len(animals_in_order) > 1:
            model = perOrderModels[animal_order]
            animal = model.predict(file)
            if animal != None:
                return animal.title()
        elif len(animals_in_order) == 1:
            return animals_in_order[0]
    return None

@app.get("/animalTypes")
def get_animal_types(pet: bool = False):
     if pet:
          return sorted(static_pet_types)
     return sorted(animal_types)