## Lawrence Livermore National Lab Self Supervised Learning Repo
This is source related to [our self-supervised paper](https://arxiv.org/abs/1711.06379) appearing in CVPR 2018. For more information take a look at the **Github Wiki page**. Just click on the [Wiki tab](https://github.com/LLNL/selfsupervised/wiki).
### What you Need
These scripts are a wrapper around Caffe. So, you need Caffe. Once you've installed it, you will probably have everything you need to run this source. These include Caffe itself, Python 2.x :snake: , OpenCV, Numpy and Pyplot. 
### What the Code Does
The source code will do the self-supervised training. The patches have been preprocessed for you. You can take a peak in https://gdo-datasci.llnl.gov/selfsupervised/download/image_sets/patches/ . Here are patches with and without chroma blurring. The training and testing lists are in Numpy/Pickle format to save space. 
### How to Run
:one: Edit **python/RunTripleContextTrain.py** and set paths in *project_home*, *path_prefix* and *gpus*. These are the only ones you must edit. Set *gpus* based on the number you want to run. In a single GPU environment, just set it to [1]. You can choose different network models that are in the *caffe_data* directory. Each one is named train_val_*network*_triple.caffenet. 

:two: Run **python/RunTripleContextTrain.py**. The first time you run it, it will create a copy in a project directory. If you are in a cluster environment, you can run the Slurm script instead. 

:three: Run the script it just created in **py_proj**. If the paths are correct, thing should run and it should start to train. Take as look at the Wiki. This has cool pictures of how it should look if it's training correctly.

:four: The figures it creates should also be saved in the **figures** directory.

:five: The trained caffe models will be saved in py_proj/*my_project*/caffe_model
### How to Test
#### CUB and CompCars
We have provided the Cub birds :hatched_chick: and Comp Cars :car: data processed as we used it. These are in: https://gdo-datasci.llnl.gov/selfsupervised/download/image_sets/ . The network models are in *caffe_data* named train_val_*network*_single.caffenet.  
#### VOC Tests
You should be able to plug the trained model into *Detection* and *Classification* tests without too much difficulty. 

:one: Try classification with an initial learning rate of 0.000025

:two: Try detection with an initial learning rate of 0.0005

:three: Segmentation requires a few tweaks. We cannot transfer fc6 and fc7 via surgery. So we have to init them when we run segmentation. By default, the *fcn.berkeleyvision.org* would not init these layers. You still need to run surgery even though you are not transferring fc6 and fc7. You can download a model that does this from https://gdo-datasci.llnl.gov/selfsupervised/download/models/caffenet/ . Notice this also has a bunch of pretrained models you can try out.  
#### Linear Tests
This runs pretty straight forward. In https://gdo-datasci.llnl.gov/selfsupervised/download/models/caffenet/ is variant linear training network that uses CaffeNet. It also chops off calls to python layers so it can be run in multi GPU mode. Other than that, it's the same as the default model. 
### Design Concepts
The source has several design choices which determine why certain things were done this way or that. 

:one: The Python source wraps around Caffe rather than using python layers. This is because python layers in Caffe do not support multiple GPU execution. This allows us preprocess data and even use a multi

:two: We are running in a cluster environment. This is why it creates a project copy when you run it. This also keeps experiments in their own tidy folder. You can look beck to check how something actually ran. 

