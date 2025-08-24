import torch
import os
import argparse
import yaml

from models import JASTNet
import utils


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Isolated Sign Language Recognition", add_help=False
    )
    parser.add_argument(
        "cfg_path",
        type=str,
        help="Path to config file",
    )
    parser.add_argument(
        "--device", default="cpu", help="device to use for training / testing"
    )

    return parser


def main(args, cfg):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = JASTNet(**cfg["model"])
    model = model.to(device)
    n_parameters = utils.count_model_parameters(model)

    print(f"Number of parameters: {n_parameters}")

    input_shape = (
        1,
        cfg.get("data", {}).get("max_sequence_length", 64),
        len(cfg["data"]["keypoints_index"]),
        2,
    )
    model_info = utils.get_model_info(model, input_shape, device)

    print("Model Information:")
    print(f"  Total parameters: {model_info['total_params']:,}")
    print(f"  Trainable parameters: {model_info['trainable_params']:,}")
    print(f"  Non-trainable parameters: {model_info['non_trainable_params']:,}")

    if "flops" in model_info:
        print(f"  FLOPs: {model_info['flops_str']}")
        print(f"  MACs: {model_info['macs_str']}")
        print(f"  Parameters (from thop): {model_info['params_str']}")
    print()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(
        "Isolated Sign Language Recognition Cal Params", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    with open(args.cfg_path, "r+", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(args, config)
