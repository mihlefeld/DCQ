# Setup

Build the docker container, this also builds the code:

`docker build -t dcq:latest .`

Use the run_all.py script to batch process multiple images:

```
docker run -it dcq:latest
python3 run_all.py -i <input_path> -o <output_path> -p <palette_size> -m <dither_mode> 
```

There are more options available:
```
Options:
  -i, --input_dir PATH
  -o, --output_dir PATH
  -k, --kernel_size INTEGER       Specify the kernel size used by the DCQ
                                  approach.
  -p, --palette_number INTEGER    Specify the maximum number of colors used
                                  for quantization
  -d, --dcq_path PATH             Specify the path to the DCQ executable.
  -m, --mode [DCQ|FS|FS-ICM|FS-ICM-DCQ|FS-DCQ]
                                  Specify which dithering algorithm to use.
  -a, --alpha_mode [divide|quantize|none]
                                  Specify how the program should handle the
                                  alpha channel.
  -c, --cluster_mode [k_means|median_cuts]
                                  Specify the clustering algorithm to
                                  initialize the palette for Floyd Steinberg
                                  Dithering
  --help                          Show this message and exit.
```

## Code Structure
### algorithm.cpp
Contains the core functionality for the dithered color quantization approach.
### alpha.cpp
Contains code to handle alpha channel pre- and post-processing.
### floyd_steinberg.cpp
Contains the dithering code for the floyd steinberg algorith.
### init.cpp
Contains codee for initializing all parameters for the DCQ approach.
### PBar.cpp
Simple progress bar code to track time passed and current progress.
### utils.cpp
Small utility functions used in multiple locations throughout the code.
