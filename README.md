# Model-Minorities
Repository for Multimodal Machine Learning (11-777) team Model Citizens

## Setup

### Data
1. Initalize the submodules to download the NLVR2 dataset
```bash
git submodule update --init
```

2. Download the images from [https://lil.nlp.cornell.edu/resources/NLVR2/](https://lil.nlp.cornell.edu/resources/NLVR2/) and extract them into the `data` directory. After this, your directory structure should look like this:
```
Model-Minorities/
└── data/
    ├── dev/
    ├── images/
    │   └── train/
    ├── test1/
    └── nlvr/
```

### Environment
1. Create a new python 3.11 virtual environment with uv
```bash
uv venv -p 3.11
```

2. Install the required packages
```bash
uv pip install -r requirements.txt
```

## Usage

To load the NLVR dataset, import and call the load_nlvr() function:
```python
from load_nlvr import load_nlvr

train_df, val_df, test_df = load_nlvr()
# Now you can use the DataFrames as needed.
```

If you want to use load_nlvr from a directory inside the project, you can use the following import:
```python
import sys

sys.path.append('..')

from load_nlvr import load_nlvr
```

