## Lawrence Livermore National Lab Self Supervised Learning Repo
This is source related to [our self-supervised paper](https://arxiv.org/abs/1711.06379) appearing in CVPR 2018. For more information take a look at the **Github Wiki page**. Just click on the Wiki tab.
### What you Need
These scripts are a wrapper around Caffe. So, you need Caffe. Once you've installed it, you will probably have everything you need to run this source. These include Caffe itself, Python 2.x, OpenCV, Numpy and Pyplot. 
### What the Code Does
The source code will do the self-supervised training. The patches have been preprocessed for you. You can take a peak in https://gdo-datasci.llnl.gov/selfsupervised/download/image_sets/patches/ . Here are patches with and without chroma blurring. The training and testing lists are in Numpy/Pickle format to save space. 
#### How to Run
1. Edit **python/RunTripleContextTrain.py** and set paths in *project_home*, *path_prefix* and *gpus*. These are the only ones you must edit. Set *gpus* based on the number you want to run. In a single GPU environment, just set it to [1]. You can choose different models that are in the *caffe_data* directory. Each one is named train_val_*network*_triple.caffenet. 
..1. Run **python/RunTripleContextTrain.py**. The first time you run it, it will create a copy in a project directory. If you are in a cluster environment, you can run the Slurm script instead. 
..1. Run the script it just created in **py_proj**. If the paths are correct, thing should run and it should start to train. Take as look at the Wiki. This has cool pictures of how it should look if it's training correctly.
..1. The figures it creates should also be saved in the **figures** directory.
..1. The trained caffe models will be saved in py_proj/*my_project*/caffe_model
#### How to Test
We have provided the Cub birds and Comp Cars data processed as we used it. These are in: https://gdo-datasci.llnl.gov/selfsupervised/download/image_sets/ . The 
#### Design Concepts
The source has several design choices which determine why certain things were done this way or that. 
1. The Python source wraps around Caffe rather than using python layers. This is because python layers in Caffe do not support multiple GPU execution. This allows us preprocess data and even use a multi precess prefetching loader.
..1. We are running in a cluster environment. This is why it creates a project copy when you run it. This also keeps experiments in their own tidy folder. You can look beck to check how something actually ran. 
