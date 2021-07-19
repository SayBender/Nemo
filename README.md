**NEmo: A wildfire smoke detection benchmark**
========
PyTorch training code and pretrained models for **NEmo** (**NE**vada s**mo**ke detection benchmark).


Facebook’s DETR model is a state of the art model created for object detection. It has a CNN backbone along with encoder-decoder transformers. It also uses a loss function for bipartite matching. Because of its architecture, it is a much simpler approach. The initial DETR model was trained using the renown COCO dataset. The DETR model showed comparable results and worked exceptionally well with larger images. 

The Nemo DETR model is a pretrained model that aims to detect wildfire in the early stages; incipient and growth stage. It is trained using 2856 unique smoke images. The validation dataset included 174 smoke images as well. 

The Nemo model ran with 5 queries, 6 layers, and 30 epochs. The results are as follows:

(put in a table)

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IoU</th>
      <th>Area</th>
      <th>AP</th>
    </tr>
   </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.50</td>
      <td>all</td>
     <td>0.031</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.50</td>
      <td>all</td>
      <td>0.074</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.75</td>
      <td>all</td>
      <td>0.020</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.50</td>
      <td>small</td>
      <td>0.059</td>
    </tr>
   <tr>
      <th>4</th>
      <td>0.50</td>
      <td>medium</td>
      <td>0.027</td>
    </tr>
   <tr>
      <th>5</th>
      <td>0.50</td>
      <td>large</td>
      <td>0.043</td>
    </tr>
  </tbody>
</table>

In this read.me file, you can find the instructions for the following tasks:
- Data Preparation 
- Testing our model 
- Fine Tuning our model 

In order to start reproducing our work, you need to install the following libraries: 

Conda Installs:
```

conda install pytorch torchvision   **Different depending on your computer**
**pytorch and torchvision must be compatible with each other, if you cannot get them to be compatible in your current environment, create a new one.**
pip install pycocotools

```

## Data preparation:

The DETR model only accepts files in COCO JSON format. In order to create a dataset, annotate your images using any tool that specializes in object labelling, or that creates bounding boxes, that will export your annotations into JSON files. If you wish to use tools that don't export directly into COCO JSON format, you can convert your YOLO or CSV files into JSON files. It is easier to combine annotations and datasets using a non-JSON format if you are working with multiple people on one training dataset. We converted our YOLO annotations into JSON files using https://github.com/Taeyoung96/Yolo-to-COCO-format-converter. If you follow the linked Github or export your annotations as a JSON file, your dataset will be ready to train the model.

## Testing the model:

In order to test the model, you need the following three things. The test.py script, our model, and a customs dataset. 

 1. Test.py script: This script can be found in our repository. Once you have downloaded this, it is important to note that all the changeable parameters such as number of layers and queries must match that of the model. If you plan on testing our model, you don't have to change anything, however, if you are testing using a model you designed with different parameters, you will need to change the code in the test.py script to match it. For example, if you trained the model with 5 queries, you need to change the default number of queries in the test.py script to 5 as well. 
 2. Model: Our model can be found in the outputs section. The model is stored under the file ‘checkpoint.pth’. 
 3. Custom dataset: In order to test the model, you will need to have a dataset consisting of images with both smoke and non-smoke data. If you don’t have a dataset of your own, we have also provided the link to Govil et al.’s fuego dataset. (https://drive.google.com/file/d/1x4bzhH-ZgEUUvh45EStFNrTT_wLSPC35/view) 

After you have all the necessary files listed above, you are ready to test the model using a custom dataset. In order to test the model, you can simply run the following script: 

```
python test.py --data_path [path_to_dataset] --resume [path_to_checkpoint.pth]
```

Here is a sample of the output: 

![Sample test result](sample_test_result.png)

If you do not achieve such a result, make sure to go back and double check the checklist. 


 ## How to finetune our model:
 
 In order to finetune our model, or completely train a new one, you can follow the following steps:

1. Clone our github repository on your local desktop. 
   - In order to clone the repository, go into your desired directory and paste the following link via terminal: https://github.com/SayBender/Nemo.git 
   - This should automatically download all the necessary files along with samples of the dataset and JSON files into your local directory
2. Make sure your dataset is in COCO JSON format. In order to do this, you can check out the data preparation section above. 
3. Next, check out the following files as you will need to make edits: smoke.py and main.py 
   - All the necessary changes are already commented in the files themselves if you are training for smoke with 4 different classes of smoke.
   - If you want to change the number of classes, go to detr.py in the models folder, then scroll down to class build(args). Change num_classes for the dataset you want to change.
     - Our 4 different classes are smoke, fire, flame, and NightSmoke. If you want to change the labels go to test.py and change them in CLASSES.
   - Look for comments as we have already added all instructions on making the changes
   - To fine tune our model or to train a new smoke model, all you have to do is change the names of the JSON files and image folders for training and validation to access your files in the smoke.py file 
     - If you are satisfied with the default layers and number of queries then you can start training your model now and skip to 4.
   - If you wish to change the number of layers or number of queries go to main.py and scroll to # * Transformer in def get_args_parser(). There you will find and can change the number of queries and encoding and decoding layers.
     - Reflect those changes in test.py if you plan on testing your model.
4. After you are done making changes and your dataset is ready, you are ready to start training the model 
   - Start small by keeping the number of epochs to 1. 
   - Run the model using the following code: 
```
python main.py --data_path ../dataset/ --output [path_to_output_folder] --epoch 1 
```
5. After you have completed training your model, you can find the output files under the ‘output’ folder. 
   - The file names ‘checkpoint.pth’ will store your model 
6. To test your model, you can following the directions under ‘Testing your model’









