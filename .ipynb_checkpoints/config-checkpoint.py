import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--class_tag", type=str, default="S1")
    parser.add_argument("--img_path", type=str, default="./checkpoints")
    parser.add_argument("--device_id", type=int, default="4")


    return parser
