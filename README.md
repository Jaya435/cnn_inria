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

All of the data used in this project can be found at https://project.inria.fr/aerialimagelabeling/download/.

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
## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning


## Authors

* **Thomas Richmond** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
