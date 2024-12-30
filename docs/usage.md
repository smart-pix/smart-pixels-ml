# Usage Guide 

This is a guide for using the Smart Pixels ML project to train and evaluate the models on the simulated data.

## 1. Installation
Refer to [Readme](README.md) for installation instructions.

## 2. Data Collection
Download the simulated data from [zenodo](https://doi.org/10.5281/zenodo.7331128)
and [PixelAV](https://docs.google.com/document/d/1ZoqVyJOOAXhzt2egMWh3OoNJ6lWq5lNR6sjcYON4Vlo/edit?tab=t.0#heading=h.k6tyal7z5t5l)
Add other links here

Ensure the two directories Data and Labels are present.

## 3. Data Preparation

- Define the paths to the data and labels directories (look at [utils](api/utils.md) for more details)
- Configure datagenerator parameters (look at [data_generator](api/data_generator.md) for more details)
- Create training and validation datagenerators

## 4. Model Creation

- Define the model architecture and compile.
- Also look at the summary of the model to ensure it is correct.
Look at [model](api/models.md) for more details on how to do that.
For loss function see [loss](api/loss.md).

## 5. Model Training 
If everything is set up correctly, the training should start and run seamlessly.
For example:
```python
model.fit(
    x=training_generator, 
    validation_data=validation_generator, 
    epochs=200, 
    verbose=1)
```

After training, check the loss and accuracy of the model.
And save the model weights.

## 6. Model Evaluation
Initiate the model and Load the weights as
```python
model=CreateModel((13,21,2),n_filters=5,pool_size=3)
model.load_weights("model_weights.h5")
```
And then evaluate the model as done in [evaluate.py](api/evaluate.md) or in [evaluate.py](../evaluate.py)

## 7. Model Prediction
Look at [predict](api/predict.md)


## 8. Add Additional Instructions
Here are some additional instructions