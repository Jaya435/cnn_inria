# A Simple Convolutional Neural Network

This project will help the user install the prerequisites neccessary to install a working CNN on their machine. The CNN here has been built using pytorch and the correct version is needed to ensure it works correctly. The CNN has been optimised for a binary classifcation, where the ground truth raster represents a max pixel value for your feature of interest.

This work has made use of the resources provided by the Edinburgh Compute and Data Facility (ECDF) (http://www.ecdf.ed.ac.uk/). The shell scripts contained within the bash directory have been created to submit jobs to the SGE engine.
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

For this to work you need Anaconda version 5.0.1. Archived Anaconda repos can be found here. Download and install the correct version for your machine.

https://repo.continuum.io/archive/

For the model to work correctly the user needs:
1. Torch verison 0.4.1
2. Cuda toolkit version 8.0.61

User should have a directory containing:
1. director named 'images' containing named RGB images
2. directory named 'gt' containing ground truth labels of said images, named the same.
```
train    
│
└───images
│   │   file011.tif
│   │   file012.tif
│   
└───gt
    │   file011.tif
    │   file012.tif
```

### Data

All of the data used in this project can be found at:

https://project.inria.fr/aerialimagelabeling/download/.

Emmanuel Maggiori, Yuliya Tarabalka, Guillaume Charpiat and Pierre Alliez. “Can Semantic Labeling Methods Generalize to Any City? The Inria Aerial Image Labeling Benchmark”. IEEE International Geoscience and Remote Sensing Symposium (IGARSS). 2017.

### Installing

A step by step series of examples that tell you how to get a development env running

Clone this repository:

```
git clone https://github.com/s1217815-ed-19/cnn_inria.git
```

Use the two specification files to ensure the correct modules are installed in each environment:

```
conda create --name mypytorch --file mypytorch-spec-file.txt
conda create --name gdal --file gdal-spec-file.txt
```
By ensuring the correct version of Anaconda is installed and then subsequently creating the two environments, the code should have the correct requirements to be run. 

## Running the model

CNN is trained end to end using the directory containing the RGB and ground truth images. Scripts within split the dataset into train, validation and test sets. Outputs figures depicting how training has evolved over time, model is saved after each iteration and stored in a directory that the user defines.  

### Stage One

This is to create a list of combinations that is then read by a shell script during model implementation. Edit the script to explore different combinations.

```
python python/grid_search.py
```

### Stage Two

Edit the final line of /bash/model_train.sh script to point to your directory containing your training dataset. 

```
python ../python/ConvNet.py --path <path/to/your/train/dir/> --arch_size "$1" --lr "$2" --batch_size "$3" --model_dict <path/to/where/you/want/to/save/modelstate.pt>


usage: ConvNet.py [-h] [--path PATH] [--batch_size BATCH_SIZE] [--lr LR]
                  [--num_epochs NUM_EPOCHS] [--arch_size ARCH_SIZE]
                  [--model_dict MODEL_DICT]

Main script to implement the CNN

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           path to train directory
  --batch_size BATCH_SIZE
                        select batch size
  --lr LR               learning rate for optimizer
  --num_epochs NUM_EPOCHS
                        Number of epochs
  --arch_size ARCH_SIZE
                        inital depth of convolution
  --model_dict MODEL_DICT
                        Path to where model is saved, extension should be .pt

```
### Stage Three
Edit the bash/cnn_accuracy.sh to point to the correct directory, then run script to determine which model is the most accurate.
```
python ${HOME}/python/accuracy.py --out_dir ${WORKING_DIR} --model_path ${WORKING_DIR}

usage: accuracy.py [-h] [--path PATH] [--model_path MODEL_PATH]
                   [--out_dir OUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           path to train directory
  --model_path MODEL_PATH
                        path to saved models
  --out_dir OUT_DIR     path to results directory
```
### Stage Four
Edit the bash/Predict-Image.sh to point to the model that produces the highest accuracy as determine in Stage Three, then run script to predict on the one image from the train dataset using that saved model in order to quality to check the results.
```
python ~/python/predict_compare.py -model /path/to/best/model -inpfile /path/to/rgb/raster -out_dir /path/to/save/results -mask /path/to/gt/raster


usage: predict_compare.py [-h] [-model MODEL] [-inpfile INPFILE] [-mask MASK]
                          [-out_dir OUT_DIR]

Predict the class of each pixel for an image and save the result. Images taken
from train folder and include mask

optional arguments:
  -h, --help        show this help message and exit
  -model MODEL      A saved pytorch model
  -inpfile INPFILE  Path and filename of image to be classified
  -mask MASK        Path to mask in train folder
  -out_dir OUT_DIR  Path to output directory
```
### Stage Five
Edit the bash/Predict-Raster.sh to point to the model that produces the highest accuracy as determine in Stage Three, then run script to predict all the images from the train or test dataset using that saved model.
```
python ${HOME}/python/raster_predict.py -model /path/to/best/model -inpfile path/to/rgb/raster -out_dir path/to/save/results

usage: raster_predict.py [-h] [-model MODEL] [-inpfile INPFILE]
                         [-out_dir OUT_DIR]

Predict the class of each pixel for an image and save the result. Images taken
from train folder.

optional arguments:
  -h, --help        show this help message and exit
  -model MODEL      A saved pytorch model
  -inpfile INPFILE  Path and filename of image to be classified
  -out_dir OUT_DIR  Output directory
```


## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [PyTorch](https://pytorch.org/get-started/locally/ - The framework used to construct CNN

## Authors

* **Thomas Richmond** - *Initial work* - [cnn_inria](https://github.com/s1217815-ed-19/cnn_inria)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thanks to Levi for helping build the inital framework
* Inspire by a desire to simplify Land Use Classification for Remote Sensing Scientists.
