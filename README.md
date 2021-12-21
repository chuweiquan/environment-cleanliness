# Environment Cleanliness
Dataset from [https://www.kaggle.com/mfadliramadhan/cleandirtygarbage](https://www.kaggle.com/mfadliramadhan/cleandirtygarbage)

### Context  
Currently, there are many people who do not care about the cleanliness of the environment. Even in their own environment sometimes the unsightly trash is still ignored. We must start caring about the cleanliness of our environment by always throwing garbage in its place. This dataset was created for the environmental cleanliness classification system, where the system will detect which environment is clean and which environment is dirty.

__Inspiration__: Environmental cleanliness is something that must be considered. That way, by keeping the environment clean, we can protect our earth and keep it clean and healthy.

## Approach
1. Importing all necessary libraries
2. The data that was provided was messy and hence needed to be cleaned and sorted into the respective training and testing datasets. We first store the dictionary paths into a dictionary, then using a loop to apply the labels to the images for both train and test sets
3. We also defined some image augmentation and callbacks. 
    1. Image augmentation: rotation, zoom, flip, translation. We do image augmentation here because our training sample is small with only 1200 training samples. However this also causes the results to vary by a huge margin for certain models.
    2. Callbacks: early stopping when loss does not appear to decrease for 3 epochs then it will terminate. Learning rate is reduced when it has stopped moving after 3 epochs. Best model is saved as model.h5
4. Depending on image size specified, we will resize all images and define the input size accordingly. The default image size is 192x192
5. A pretrained model will be loaded depending on the model specified. The default model is vgg16
6. With the data preparation and pretrain model loaded, we build our model
    
```python
Sequential([ 
    data_augmentation,
    model,

    Flatten(),
    Dense(512, activation = 'relu'),
    Dense(256, activation = 'relu'),
    BatchNormalization(),
    Dropout(0.25),

    Dense(experiment.num_classes, activation = 'sigmoid')
])
```
    
7. We train the model with adam optimizer with a learning rate of 0.0001, loss function as “binary_crossentropy” as it is a binary classification problem, and metrics that we look out for is accuracy. We train the model on 30 epochs and a batch size of 32
8. We retrain the model after unfreezing the bottom convolution layers for the pretrained model and evaluate the new results

## Code Execution
1. Create virtual environment with virtualenv
2. run requirements.txt
3. In the terminal, run `python rubbish_detectionv2.py --model <your model> --imgsize <your image size> -- unfreeze <True/False> --layers <num layers depending on model>`
    1. If unfreeze is False, then it will not unfreeze the model and retrain the model. 
    2. If unfreeze is True, num layers must be specified otherwise the model will be retrained with fully freezed layers

## Results
| Model | Loss | Accuracy | Loss (fine-tune) | Accuracy (fine-tune) | Total Layers | Untrained Layers |
| --- | --- | --- | --- | --- | --- | --- |
| VGG16 | 0.491 | 0.738 | 0.252 | 0.888 | 19 | 11 |
| ResNet50 | 1.318 | 0.5 | 0.744 | 0.568 | 175 | 143 |
| ResNet50v2 | 0.404 | 0.825 | 0.421 | 0.846 | 175 | 139 |
| DenseNet201 | 0.387 | 0.832 | 0.361 | 0.858 | 427 | 315 |
| Xception | 0.444 | 0.8 | 0.284 | 0.889 | 132 | 106 |

## Areas of Improvement
- Test out different augmentations to find out whether the augmentations are helping or harming the model’s learning
- Test out different learning rate and different optimizers
- Test out different configurations of the FCNN
- Explore using OpenCV to facilitate training processing
    - HSV values that are higher for detecting rubbish
    - Edge detection for rubbish (revisit)

