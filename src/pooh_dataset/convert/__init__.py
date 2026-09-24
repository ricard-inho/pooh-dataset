"""rosbag -> Hub dataset conversion (maintainers; ``pip install pooh-dataset[convert]``).

TODO: move ``bag_to_dataset/convert.py`` here. Output must keep the layout the reader
expects (``<modality>/<trajectory>/*.parquet``, ``calibration/<trajectory>.yaml``,
``manifest.json``); see DATA_REQUIREMENTS.md.
"""
