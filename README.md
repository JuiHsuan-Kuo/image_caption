# image_caption

This is a simple implementation of image caption, estimator_version is another implementation which is based on tf.Estimator 

## Installation
1. python 3.6
2. tensorflow (>=1.4)
3. Keras
4. h5py

## simple step
1. git clone  

2. download coco image caption annotations and images from [`here`](http://cocodataset.org/#download). name `annotations` and `images` respectively and put them in the same directory.

3. cd image_caption 

4. create symbolic link
```
ln -s /path/to/data ./data
```
5. do the preprocessing
```
python prepro.py
```
6. for training, please specify the parameters in `train.sh` and  
```
sh train.sh
```

7. for inference, please specify the image path in `inference.sh` and  
```
sh inference.sh
```
it will produce a caption of the image  

