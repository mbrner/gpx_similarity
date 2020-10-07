# GPX Similartity

This projects tries to compare gpx routes (gpx files) and find similar segments in different routes. To compare the routes the routes are converted in a series of images via OpenStreetMap. The images are embedded using a neural network model. The distance between the embedded images is the 'similarity'.

Via this scripts all necessary steps can be run to build up a training database, train the embedding model, build up a reference database and runs comparisons. The embedding is independent from the comparisons you want to run, therefore the training has to be done once. You can find pretrained models in the repository https://github.com/mbrner/gpx_similarity/example.

To start a step `[python] gpx_similarity.py CMD`.

The order of CMDs are:

1. add-train-files

2. train-model

This produces a trained model. A trained model is needed to build up a reference database against with comparisons can be executed.

3. add-reference-files

4. compare-gpx

There a more command you can run. Check the help `python gpx_similarity.py CMD --help` to learn about their purpose.

DISCLAIMER: I don't claim that the approach I follow in this project is a good/the best approach. For me this was more about the developement process.


<!--- HIDE IN CLICK --->
## Install

I recommend to use one of the environment.yaml files to create a conda env. In the created env this project can be installed via 
```pip install .```.
After installation the CLI of the project can be used via `$ gpx_similarity`.

## Usage

To try the code simple go to the example: https://github.com/mbrner/gpx_similarity/example

The actual comparison of an gpx file with a database of stored files starts a Dash+Plotly webserver. A more detailed explanation of the visualization is shown at the bottom of the Plotly site.

### Screenshots

#### Raw images of the most similar segements
![Visualization raw segments](./images/screen_1.png)

#### Encoded+decoded version of the most similar segments
![Visualization embedded segments](./images/screen_2.png)

## Training

As stated in the first section, the comparison of segments is done in the latent space of an convolutional autoencoder. The changing embedding of the segments images looks like this:


![Training process](./images/training_process.gif)

