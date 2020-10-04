# GPX Similartity

This projects tries to compare gpx routes (gpx files) and find similar segments in different routes. To compare the routes the routes are converted in a series of images via OpenStreetMap. The images are embedded using a neural network model. The distance between the embedded images is the 'similarity'.

Via this scripts all necessary steps can be run to build up a training database, train the embedding model, build up a reference database and runs comparisons. The embedding is independent from the comparisons you want to run, therefore the training has to be done once. You can find pretrained models in the repository https://github.com/mbrner/gpx_similarity.

To start a step `python gpx_similarity.py CMD`

The order of CMDs are:

1. add-train-files

2. train-model

This produces a trained model. A trained model is needed to build up a reference database against with comparisons can be executed.

3. add-reference-files

4. run-comparison

There a more command you can run. Check the help `python gpx_similarity.py CMD --help` to learn about their purpose.

DISCLAIMER: I don't claim that the approach I follow in this project is a good/the best approach. For me this was more about the developement process.


< !--- HIDE IN CLICK --->

Test
