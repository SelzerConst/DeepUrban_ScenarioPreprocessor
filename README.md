## Packaged Utils from DeepScenario

Package needed to create DeepUrban Scenarios from the DeepUrban Data

Package installation
```sh
pip install -e .
```

Adjust needed configurations in config/default.yaml

## DeepUrban Dataset
The DeepUrban raw data can be downloaded from [DeepScenario](https://app.deepscenario.com/explore/release-list).
All dill scenarios are automatically created within its location folder, as well as the maps folder with all its content.
The splits for the locations can be downloaded from [DeepUrban](https://iv.ee.hm.edu/pipeline/).

Recommended folder structure see below.
```
/DeepUrban_folder/
        ├── /src_dir/
        |   ├── <location_1>/
        |   |   └── release
        |   |       ├── data
        |   |       ├── doc
        |   |       ├── scripts
        |   |       └── ...
        |   ├── <location_2>/
        |   |   └── release
        |   |       ├── data
        |   |       ├── doc
        |   |       ├── scripts
        |   |       └── ...
        |   └── ...
        ├── output_dir/
        |   ├── <location_1>/
        |   |   ├── <location_1_scenario>.dill
        |   |   ├── <location_1_scenario>.dill
        |   |   └── ...
        |   ├── <location_2>/
        |   |   ├── <location_2_scenario>.dill
        |   |   ├── <location_2_scenario>.dill
        |   |   └── ...
        |   └── ...
        ├── maps/
        |   ├── location_meta.json
        |   ├── <location_1_map>.osm
        |   ├── <location_2_map>.osm
        |   └── ...
        └── split/
            ├── <location_1_split>.yaml
            ├── <location_2_split>.yaml
            └── ...
```