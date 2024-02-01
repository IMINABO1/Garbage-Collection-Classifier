# Recycler
Recycler is an AI model that selects what kind of trash an object is. It is a Garbage-Collection-Classifier

The AI was trained using Xception model. The wasn't much preprocessing of input image since the dataset was not extremely large.

The Ai model takes in an input image and predicts what category of trash it belongs to. 
It can be used when selecting what recycle bin to discard an item.

The `Classifier.ipynb` is the Jupyternotebook used to train the model.\
The `predict.ipynb` is the Jupyternotebook used for prediction of just one image.\
The `categorize.py`is the python file to categorize the trash using the AI model `model.h5`\
The `server.py` is a Flask app that would be connected to a web application

