# `gpx_similarity` Example
## Apply Model and Run Comparisons
With the files in this folder the application of a trained model and the comparison of gpx files can be tested.

To run the following commands unpack the model checkpoint from `final_model.ckpt.tar.xz`. All infos from the training process can be found in the `full_training_logs.tar.xz` archive. Unpack the archive and run `tensorboard --logdir <extract_path>/logs`

 In additionally a redis server must be running. The easiest way to run redist is to use the [docker-compose-file](../docker-compose.yaml) in the root directory of this repo. In cases a different instance should be used the redis connection infos in the `config.toml` must be adjusted.

To build a reference database, call:
```gpx_similarity add-reference-files -d komoot config.toml example.db final_model.ckpt gpx_in_database/*.gpx```
This commands results in the example database `example.db` also present in this repository. 

To run a comparison and start the visualization server, call:
```gpx_similarity compare-gpx config.toml example.db final_model.ckpt gpx_test/2020-04-05_166924972_cleaned.gpx```

## Train own model
To train a model gpx files are needed. For this example I trained a model based on gpx files form the 
[GeoLife GPS Trajectories](https://www.microsoft.com/en-us/download/details.aspx?id=52367) and bikepacking routes from www.bikepacking.com. The repository contains [utility](../utils) scripts to scrape gpx files from bikepacking.com and convert the geolife dataset to gpx files.

The first step to train a model is to add gpx files to the training database. The training process loads preprocessed images from a postgres database. The easiest way to run a postgres instance is to use the [docker-compose-file](../docker-compose.yaml) in the root directory of this repo. In cases a different instance should be used the postgres credentials in the `config.toml` must be adjusted.

To add images to the database run:
```gpx_similarity add-train-files config.toml <folder>/*.gpx```
Check `gpx_similarity add-train-files --help` for more information.

To train a model 
```gpx_similarity train-model config.toml <output_dir>```
Check `gpx_similarity train-model --help` for more information. During the training process, the model graph training/test loss and some test segements can be accessed via `tensorboard --logidr <output_dir>/logs`. After each epoch the model is saved as `<output_dir>/<datetime_str>/logs/models/cp-<epoch_num>.ckpt`.
