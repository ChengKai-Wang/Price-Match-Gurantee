# Price-Match-Guarantee

## Introduction
1. This is a [Kaggle Competition](https://www.kaggle.com/c/shopee-product-matching).
2. The aim of this competition is match the same products through the images or descriptions of the items. 
3. I use **Momentum Contrast** (K. He, *CVPR* 2020) as model architecture to learn the image representations.
![](https://i.imgur.com/AwGuvca.png)
4. Dataset and generated dataset falls under Shopee Terms and Conditions which can be seen on [Kaggle Datasets](https://www.kaggle.com/c/shopee-product-matching/data)
## Experiments
* Experiment flow (I only access image features)
![](https://i.imgur.com/HU0LnWw.png)
* Parameters Setting
    * CNN model: ResNet18
    * Embedding length: 128
    * K(dictionary size): 1024
    * M(momentum): 0.99
    * T(temperature): 0.07
    * Epochs: 20
* F1 score and threshold testing on test set
![](https://i.imgur.com/bRt5etH.png)
* t-SNE on the maximum 9 groups (the points in same color are the same products)
![](https://i.imgur.com/rB9hkDs.png)

## Results
![](https://i.imgur.com/2df55CO.png)

## Guide
### Environment Requirments
* python version >= 3.6


| Package Name | Version |
| ------------ | ------- |
| pandas       | 1.3.2   |
| numpy        | 1.20.3  |
| pytorch      | 1.9.0   |
| scikit-learn | 1.0.1   |
| tqdm         | 4.62.1  |
| pillow       | 8.0.0   |
| torchvision  | 0.10.0  |
| matplotlib   | 3.4.3   |

### Reproducibility
* Run **MoCo.py**
    * This file is for training
    * The models will be saved in "models" folder and the training messages will be saved in "log.txt"
* Run **test.py**
    * This file is for testing
    * The recall-precision curve will be saved in "result.png"




