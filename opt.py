import torch
import torch.nn.functional as F

from utils import accuracy, top_k_accuracy
from logger import MetricLogger, SmoothedValue


def train_one_epoch(
    args, model, data_loader, optimizer, epoch, print_freq=10, log_dir="logs"
):
    model.train()

    metric_logger = MetricLogger(delimiter="  ", log_dir=log_dir)
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Train epoch: [{epoch}]"

    for param_group in optimizer.param_groups:
        metric_logger.update(lr=param_group["lr"])

    for batch_idx, batch in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        joint_coords = batch["keypoints"].to(args.device)
        targets = batch["labels"].to(args.device)

        logits = model(joint_coords)

        loss = F.cross_entropy(logits, targets)

        overall_accuracy = accuracy(logits, targets)
        top5_accuracy = top_k_accuracy(logits, targets, k=5)

        metric_logger.update(acc=overall_accuracy)
        metric_logger.update(top5_acc=top5_accuracy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    print(f"Train stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_fn(
    args, data_loader, model, epoch, print_freq=100, results_path=None, log_dir="logs"
):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ", log_dir=log_dir)
    header = f"Test: [{epoch}]"

    all_predictions = []
    all_targets = []
    all_logits = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            joint_coords = batch["keypoints"].to(args.device)
            targets = batch["labels"].to(args.device)

            logits = model(joint_coords)

            loss = F.cross_entropy(logits, targets)

            pred_labels = torch.argmax(logits, dim=1)

            metric_logger.update(loss=loss.item())

            all_predictions.extend(pred_labels.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_logits.append(logits.cpu())

    if all_logits:
        all_logits_tensor = torch.cat(all_logits, dim=0)
        all_targets_tensor = torch.tensor(all_targets, device=all_logits_tensor.device)

        overall_accuracy = accuracy(all_logits_tensor, all_targets_tensor)
        top5_accuracy = top_k_accuracy(all_logits_tensor, all_targets_tensor, k=5)

        metric_logger.update(acc=overall_accuracy)
        metric_logger.update(top5_acc=top5_accuracy)

    metric_logger.synchronize_between_processes()
    print(f"Test stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate_one_epoch(args, model, data_loader, epoch, print_freq=50, log_dir="logs"):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ", log_dir=log_dir)
    header = f"Validation epoch: [{epoch}]"

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            joint_coords = batch["keypoints"].to(args.device)
            targets = batch["labels"].to(args.device)

            logits = model(joint_coords)

            loss = F.cross_entropy(logits, targets)

            metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    print(f"Validation stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
