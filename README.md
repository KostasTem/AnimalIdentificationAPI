# AnimalIdentificationAPI

This project is an API made using the FastAPI framework that is used to identify animals from pictures with machine learning using Tensorflow and the ResNet50 model.

The identification process happens in two steps. First the picture is run through a model that predicts the order the animal in the picture belongs to. Orders are the families animals belong to, meaning animals of the same order look similar to each other. After the order is predicted, the picture is ran through a model that has been trained on that order.

## Training
To train the TopLevelClassifier you need to create a folder named TopLevelClassfierData which contains a folder for each order displayed in the animal_types.txt with each one of those folders containing all the images of the animals in that order.
```python
from animal_classification import AnimalClassificationModel
topLevelModel = AnimalClassificationModel("TopLevelClassifier", "TopLevelClassifierData", True)
topLevelModel.train()
```
To train the classifiers for each order you need to make a folder named PerOrderClassifierData, add folders for each order inside it, add a folder for each animal contained in each order in the corresponding folders which contain images of each animal.
```python
for order in topLevelModel.classes:
     perOrderModels[order] = AnimalClassificationModel(f"{order.title()}Classifier", f"PerOrderClassifierData/{order}", True)
     perOrderModels[order].train()
```
