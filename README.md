**NEmo: A wildfire smoke detection benchmark**
========
PyTorch training code and pretrained models for **NEmo** (**NE**vada s**mo**ke detection benchmark).

test
### Temporary instructions
> This is going to host a single standard centralized fully maintained most up to date code for our Smoke Detection. 

Connor, please put all the code cleaned and named properly (**NOT** based on WIDER face anymore) but based on our own thing. 
Make sure to complete this README file in a way that literally if I cloned this to my computer and simply followed your instructions I can replicate it.

make sure to include sections to:
- how to train a new model (plz include the training and validatoin bbox files)
Plz Dont upload the entire data to github, just placeholders and maybe 2 images and a text file mentioning where in Nevada Box and Gpuh to find the data to put in the folder, we don't use github to host large files, json and csv file are ok.  
- how to test the pre-existing models (please upload the pre-trained models in a specific folder)

All the instructions should be clear and easy to follow.

location wise, things in the repository must be in a way that works out of the box after cloning the environment (the user should only need to copy the dataset to where it wants in the server or local directory) everything else is provided (including all the bounding boxes for training and validating the model. Create the repository and its readme file in a way that you could use it to set everything up in another computer and actually test that it works.

For your computing environment, you can provide the spec-list of the conda environment that you used to run the successful final experiments.
under your conda environment do this command
```
conda list --explicit > spec-file.txt
```


Christina and Yongyi can also help write a section about the data preparation.

## Data Preparations:
The DETR model only accepts JSON files. So you can annotate with any tool that creates bounding boxes that will export your annotations into JSON files or convert your annotations into JSON. It is easier to combine annotaions and datasets using a non-JSON format if you are working with multiple people on one training dataset. We converted our YOLO annotations into JSON files using https://github.com/Taeyoung96/Yolo-to-COCO-format-converter. If you follow the linked Github or export your annotations as a JSON file, your dataset will be ready to train the model.

To create an organized and professional looking README for the codes and models that you upload to this repository, you can get inspired by the readme file of other repositories or cheatsheets online about github text editing.

## **This REPOSITORY is Private** 
