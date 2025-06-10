# m2p2_parser
Offline Data Processor for M2P2 dataset 

## Run

``` 
python3 m2p2_parser.py --folder /path/to/your/rosbags --file /path/to/output_data_directory

-b, --folder: Path to the directory containing your ROS bag files.

-f, --file: Path to the directory where the output .pkl files and image folders will be saved.
```

The script generates the output directory with images and pickle files. 
