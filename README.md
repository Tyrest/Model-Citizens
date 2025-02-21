# Model-Minorities
Repository for Multimodal Machine Learning (11-777) team Model Citizens

## Setup
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
