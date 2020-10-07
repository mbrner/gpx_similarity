


`gpx_similarity add-reference-files -d komoot config.toml example.db trained_model/model_epoch_009.weights/model_epoch_009.weights gpx_in_database/*.gpx`


gpx_similarity compare-gpx config.toml example.db trained_model/model_epoch_009.weights/model_epoch_009.weights gpx_test/2020-04-05_166924972_cleaned.gpx`