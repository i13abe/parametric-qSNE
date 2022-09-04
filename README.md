# parametric q-SNE
Parametric q-Gaussian distributed stochastic neighbor embedding with Convolutional Neural Network

The parametric q-SNE is a non-linear parametric dimensionality reduction technique to extend q-SNE. The parametric q-SNE uses q-Gaussian distribution in low-dimensional space instead of t-distribution of parametric t-SNE. The q-Gaussian distribution is a probability distribution maximized the tsallis entropy under appropriate constraints. It is generalization of Gaussian distribution with hyperparameter q. It has Gaussian distribution when q close to 1, and t-distribution when q equal to 2.


The details for thw q-SNE can be found in 'https://ieeexplore.ieee.org/document/9533781'.
This paper is accepted IJCNN2021.

In this GitHub, we provide the implementation of parametric q-SNE on Pytorch.

# Instllation
Requirements:
+ Python 3.6+
+ scikit-learn 0.23.2+
+ numpy 1.18+
+ scipy 1.5+
+ cython 0.29.24+
+ pytorch 1.7+
+ matplotlib 3.3+
+ gcc 7.5.0+ (to compile the cython file (.pyx))
+ OS Ubuntu 18.04.4


These requirments is just my development enviroment.

Please manual install to get this package:
```
git clone git@github.com:i13abe/parametric-qSNE.git
```

Install the requirements:
```
pip install -r requirements.txt
```

Install the pytorch which is suitable for your enviroment.
https://pytorch.org/

# How to use the parametric q-SNE
We provide the parametriic_qsne.ipynb to run demonstrate on MNIST.
If you can use jupyter notebook or jupter lab, please use this demonstrate file.
If you can not use jupyter, please make the python file while referring parametriic_qsne.ipynb.
Before run it, following step is required.

- compile the "_utils.pyx"
```
$ cd utils/qsne_utils
$ python setup.py build_ext --inplace
```

>If you have any error, maybe your gcc is wrong or cython version is wrong.
>When you can not compile "_util.pyx", please modify "from .qsne_utils import _utils" at 6 line in "parametric_qsne.py" to "import utils as _utils".
>We prepare the "_binary_search_perplexity" function in "utils.py" for this wrong case (however it takes long time).


- Run tips
Our implementation parametric q-SNE can use any CNN, e.g. VGG, ResNet etc.
In our parametric_qsne.ipynb, we use small CNN prepared in utils.networks.py.
If you want to use them, please rewrite the calling sentence of network.
The parametric q-SNE requires larger mini-batch size.
The larger mini-batch size, the better results we get.