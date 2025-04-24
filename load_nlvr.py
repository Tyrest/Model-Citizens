import os
import pandas as pd

DATA_DIR = os.path.join("data", "nlvr", "nlvr2", "data")
TRAIN_FILE = "train.json"
DEV_FILE = "dev.json"
TEST_FILE = "test1.json"
TEST_2_FILE = "test2.json"

# TRAIN_IMAGE_DIR = os.path.join("data", "images", "train")
# VAL_IMAGE_DIR = os.path.join("data", "dev")
# TEST_IMAGE_DIR = os.path.join("data", "test1")

TRAIN_IMAGE_DIR = "/local/data/images/train"
VAL_IMAGE_DIR = "/local/data/dev"
TEST_IMAGE_DIR = "/local/data/test1"
TEST_2_IMAGE_DIR = "/local/data/test2"


def _process_val_df(df, image_dir):
    df = df[["label", "sentence", "identifier"]].copy()
    df["left"] = df["identifier"].apply(
        lambda x: os.path.join(image_dir, "-".join(x.split("-")[:-1] + ["img0.png"]))
    )
    df["right"] = df["identifier"].apply(
        lambda x: os.path.join(image_dir, "-".join(x.split("-")[:-1] + ["img1.png"]))
    )
    return df


def _process_train_df(df):
    df = df[["label", "sentence", "identifier", "directory"]].copy()

    def get_image_path(row, suffix):
        return os.path.join(
            TRAIN_IMAGE_DIR,
            str(row["directory"]),
            "-".join(row["identifier"].split("-")[:-1] + [suffix]),
        )

    df["left"] = df.apply(lambda row: get_image_path(row, "img0.png"), axis=1)
    df["right"] = df.apply(lambda row: get_image_path(row, "img1.png"), axis=1)
    return df


def load_nlvr(return_test_2=False):
    val_df = pd.read_json(os.path.join(DATA_DIR, DEV_FILE), lines=True)
    val_df = _process_val_df(val_df, VAL_IMAGE_DIR)

    test_df = pd.read_json(os.path.join(DATA_DIR, TEST_FILE), lines=True)
    test_df = _process_val_df(test_df, TEST_IMAGE_DIR)

    train_df = pd.read_json(os.path.join(DATA_DIR, TRAIN_FILE), lines=True)
    train_df = _process_train_df(train_df)

    if return_test_2:
        test_2_df = pd.read_json(os.path.join(DATA_DIR, TEST_2_FILE), lines=True)
        test_2_df = _process_val_df(test_2_df, TEST_2_IMAGE_DIR)
        return train_df, val_df, test_df, test_2_df

    return train_df, val_df, test_df
