# ppl_keras_lab
Solution for Python Programming Language laboratory on Keras
## Setup
- The project's virtual environment is built under Python 3.8.3, as specified in the laboratory instruction. 
- All the packages are listed in the `requirements.txt` file. You should install them in your virtual environment.
- Please run `plaidml-setup` to choose the devices to run the project on. 
## Performed tasks
1. Having prior worked with the Keras package, we first run the models and then add the layers to make the comparison on the results - to the first one, a simple dense layer, whereas to the second - Maxpool, Conv2D and Dropout layers (it helps preventing overfitting). In all cases, model fast achieves high accuracy for both train and validation sets, but generally too many deep layers may be a bit overkill and lead to worsening the performance and possibly overfitting. In this case, after adding only one, the results still improved. The dataset (MNIST) is overally fairly simple, so it is to be expected, that results will be good but models could be prone to such issues. Therefore, as shown in ex.1 file, the usage of convolutional layers is also not even that necessary.  

| Model         | Baseline error |
|---            | ---            |
| Ex.1 Base     | 1.97%          |
| Ex.1 Extended | 1.94%          |
| Ex.2 Base     | 1.97%          |
| Ex.2 Extended | Text           | 
  
The loss and accuracy plots from the training are available in *results* folder.  
2. We added file saving and loading to second example file as well as `get_model` function in `Keras_PyQt_Paint_Model.py` file.  
3. 