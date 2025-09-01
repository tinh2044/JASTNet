import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import time
import argparse
import json
import datetime
import numpy as np
import yaml
import random
from pathlib import Path
from loguru import logger

from dataset import Datasets
from optimizer import build_optimizer, build_scheduler
from models import JASTNet
from opt import train_one_epoch, evaluate_fn
import utils


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Isolated Sign Language Recognition Training", add_help=False
    )
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=100, type=int)

    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument(
        "--local-rank", type=int, default=0, help="Local rank for distributed training"
    )

    parser.add_argument(
        "--device", default="cpu", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--num_workers", default=4, type=int)

    parser.add_argument(
        "--cfg_path",
        type=str,
        default="configs/LSA-64.yaml",
        help="Path to config file",
    )

    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument(
        "--save_samples", action="store_true", help="Save sample visualizations"
    )

    return parser


def setup_distributed():
    """Initialize distributed training"""
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        print(
            f"Setting up distributed training: rank={rank}, local_rank={local_rank}, world_size={world_size}"
        )

        # Initialize the process group
        dist.init_process_group(backend="nccl")

        # Set the device for this process
        torch.cuda.set_device(local_rank)

        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main(args, cfg):
    # Setup distributed training
    is_distributed, rank, local_rank, world_size = setup_distributed()

    # Only print from rank 0
    if rank != 0:
        logger.remove()
        logger.add(lambda msg: None)  # Suppress logs from other ranks

    model_dir = cfg.get("training", {}).get("model_dir", "outputs/islr_model")
    log_dir = f"{model_dir}/log"

    # Use local_rank for device selection in distributed mode
    if is_distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    seed = args.seed + rank  # Different seed for each rank
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    # Create datasets
    cfg_data = cfg.get("data", {})
    train_data = Datasets(
        cfg_data.get("root", "data/LSA-64"),
        cfg_data.get("train_dir", "train"),
        augment=True,
        keypoints_index=cfg_data.get("keypoints_index", []),
        cfg=cfg_data,
    )
    test_data = Datasets(
        cfg_data.get("root", "data/LSA-64"),
        cfg_data.get("test_dir", "test"),
        augment=False,
        keypoints_index=cfg_data.get("keypoints_index", []),
        cfg=cfg_data,
    )

    # Create distributed samplers
    train_sampler = None
    test_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(train_data, shuffle=True)
        test_sampler = DistributedSampler(test_data, shuffle=False)

    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=train_data.data_collator
        if hasattr(train_data, "data_collator")
        else None,
        shuffle=(train_sampler is None),  # Only shuffle if no sampler
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=test_data.data_collator
        if hasattr(test_data, "data_collator")
        else None,
        sampler=test_sampler,
        pin_memory=True,
    )

    # Create JASTNet model
    model = JASTNet(**cfg["model"])
    model = model.to(device)

    # Wrap model with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        # Access the actual model for parameter counting
        model_for_params = model.module
    else:
        model_for_params = model

    n_parameters = utils.count_model_parameters(model_for_params)

    if rank == 0:  # Only print from rank 0
        print(f"Number of parameters: {n_parameters}")

        # Calculate model info
        input_shape = (
            args.batch_size,
            cfg.get("data", {}).get("max_sequence_length", 64),
            len(cfg["data"]["keypoints_index"]),
            2,
        )
        model_info = utils.get_model_info(model_for_params, input_shape, device)

        print("Model Information:")
        print(f"  Total parameters: {model_info['total_params']:,}")
        print(f"  Trainable parameters: {model_info['trainable_params']:,}")
        print(f"  Non-trainable parameters: {model_info['non_trainable_params']:,}")

        if "flops" in model_info:
            print(f"  FLOPs: {model_info['flops_str']}")
            print(f"  MACs: {model_info['macs_str']}")
            print(f"  Parameters (from thop): {model_info['params_str']}")
        print()

    # Load pretrained model if specified
    if args.finetune:
        if rank == 0:
            print(f"Finetuning from {args.finetune}")
        checkpoint = torch.load(args.finetune, map_location="cpu")
        ret = model_for_params.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if rank == 0:
            print("Missing keys: \n", "\n".join(ret.missing_keys))
            print("Unexpected keys: \n", "\n".join(ret.unexpected_keys))

    # Create optimizer and scheduler
    optimizer_config = cfg.get("training", {}).get("optimization", {})
    optimizer = build_optimizer(config=optimizer_config, model=model_for_params)

    # Set initial_lr for optimizer (needed for scheduler resume)
    for group in optimizer.param_groups:
        if "initial_lr" not in group:
            group["initial_lr"] = group["lr"]

    # Update config with total epochs for warmup scheduler
    if "training" not in cfg:
        cfg["training"] = {}
    if "optimization" not in cfg["training"]:
        cfg["training"]["optimization"] = {}
    cfg["training"]["optimization"]["total_epochs"] = args.epochs

    # Initialize scheduler with correct last_epoch for resume
    scheduler_last_epoch = -1
    if args.resume:
        if rank == 0:
            print(f"Resume training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        if utils.check_state_dict(model_for_params, checkpoint["model_state_dict"]):
            ret = model_for_params.load_state_dict(
                checkpoint["model_state_dict"], strict=False
            )
        else:
            print("Model and state dict are different")
            raise ValueError("Model and state dict are different")

        if "epoch" in checkpoint:
            scheduler_last_epoch = checkpoint["epoch"]
        args.start_epoch = checkpoint["epoch"] + 1
        if rank == 0:
            print("Missing keys: \n", "\n".join(ret.missing_keys))
            print("Unexpected keys: \n", "\n".join(ret.unexpected_keys))

    # Create scheduler with correct last_epoch
    scheduler, scheduler_type = build_scheduler(
        config=cfg["training"]["optimization"],
        optimizer=optimizer,
        last_epoch=scheduler_last_epoch,
    )

    # Load optimizer and scheduler state if resuming
    if args.resume:
        if (
            not args.eval
            and "optimizer_state_dict" in checkpoint
            and "scheduler_state_dict" in checkpoint
        ):
            if rank == 0:
                print("Loading optimizer and scheduler")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if rank == 0:
                print(f"New learning rate : {scheduler.get_last_lr()[0]}")

    # Add loss function and output directory to args
    args.output_dir = model_dir
    args.save_images = cfg.get("evaluation", {}).get("save_images", False)
    args.save_samples = args.save_samples

    output_dir = Path(model_dir)

    if args.eval:
        if not args.resume:
            logger.warning(
                "Please specify the trained model: --resume /path/to/best_checkpoint.pth"
            )

        test_results = evaluate_fn(
            args,
            test_dataloader,
            model,
            epoch=0,
            print_freq=args.print_freq,
            results_path=f"{model_dir}/test_results.json",
            log_dir=f"{log_dir}/eval/test",
        )
        if rank == 0:
            print(
                f"Test acc of the network on the {len(test_dataloader)} test samples: {test_results['acc']:.3f}"
            )
            print(f"* TEST LOSS {test_results['loss']:.3f}")
        cleanup_distributed()
        return

    if rank == 0:
        print(f"Training on {device}")
        print(
            f"Start training for {args.epochs} epochs and start epoch: {args.start_epoch}"
        )
        if is_distributed:
            print(f"Distributed training with {world_size} GPUs")

    start_time = time.time()
    best_acc = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        # Set epoch for distributed sampler
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if rank == 0:
            logger.info(f"Epoch {epoch} of {args.epochs}")

        train_results = train_one_epoch(
            args,
            model,
            train_dataloader,
            optimizer,
            epoch,
            print_freq=args.print_freq,
            log_dir=f"{log_dir}/train" if rank == 0 else None,
        )
        scheduler.step()

        # Save checkpoint only from rank 0
        if rank == 0:
            checkpoint_paths = [output_dir / f"checkpoint_{epoch}.pth"]
            prev_chkpt = output_dir / f"checkpoint_{epoch - 1}.pth"
            if os.path.exists(prev_chkpt):
                os.remove(prev_chkpt)
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model_state_dict": model_for_params.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )
            logger.info("")

        # Evaluate
        test_results = evaluate_fn(
            args,
            test_dataloader,
            model,
            epoch,
            print_freq=args.print_freq,
            log_dir=f"{log_dir}/test" if rank == 0 else None,
        )

        # Save best model based on acc (only from rank 0)
        if rank == 0 and test_results["acc"] > best_acc:
            best_acc = test_results["acc"]
            checkpoint_paths = [output_dir / "best_checkpoint.pth"]
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model_state_dict": model_for_params.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    checkpoint_path,
                )

        if rank == 0:
            logger.info(f"* TEST acc {test_results['acc']:.3f} Best acc {best_acc:.3f}")

            # Log results
            log_results = {
                **{f"train_{k}": v for k, v in train_results.items()},
                **{f"test_{k}": v for k, v in test_results.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
            logger.info("")
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_results) + "\n")

    if rank == 0:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Training time {total_time_str}")

    cleanup_distributed()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser(
        "Isolated Sign Language Recognition", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    with open(args.cfg_path, "r+", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Set default model directory if not specified
    if "training" not in config:
        config["training"] = {}
    if "model_dir" not in config["training"]:
        config["training"]["model_dir"] = "outputs/islr_model"

    Path(config["training"]["model_dir"]).mkdir(parents=True, exist_ok=True)
    main(args, config)
