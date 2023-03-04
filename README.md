# CACRWNet
Context-aware and Channel re-weighting network with Noise suppression for Remote sensing scene classification

## Instruction
* `split.py`

Once you've obtained the necessary dataset, use `split.py` to proportionally partition the dataset into training and test sets. The code simply needs to change the source and new file paths and divide the ratio.

* `reshape.py`

Its role is to crop the image size to 224×224, all you have to do is alter the source file directory and the new file location, and you can also change the size of the crop image.

* `train.py`

Change the path of the training and test sets first, then the number of experiments.
 
  `num_epochs = config.NUM_EPOCHS`
  
  `num = 1`

The optimizer can also be customized to meet your needs.

* `config.py`

You can change the data set category, training batch size, number of training rounds, and path to store the model from `config.py`.

## Datasets

* UC Merced Land Use Dataset:

http://weegee.vision.ucmerced.edu/datasets/landuse.html

* AID Dataset:

https://captain-whu.github.io/AID/

* NWPU RESISC45:

http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html

## Enviroment

python = 3.8.10

torch = 1.10.0

cuda = 11.3


