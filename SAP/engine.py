import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import util.misc as utils
import torch.nn.functional as F


def sigmoid_binary_focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction=True):
    # input: [bs, num_queries, 256]
    # targets: [bs, num_queries, 256]
    # text_mask: [bs, max_num_patches,], max_num_patches < 256
    num_queries, _ = inputs.size()

    inputs = inputs.view(-1, 117)
    targets = targets.view(-1, 117)

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    accuracy = ((p > 0.5).to(targets.dtype) * targets).sum() / (targets.sum() + 1e-6)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if loss.sum() < 0:
        print(targets)
        print(alpha_t)
    if reduction:
        return loss.sum(), accuracy
    else:
        return loss.mean(), accuracy


def train_one_epoch(model: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    # criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('interaction_loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('verb_loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    for samples, qwen_pixel_values, image_grid_thw, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = [s.to(device) for s in samples]
        qwen_pixel_values = torch.cat(qwen_pixel_values, dim=0).to(device)
        image_grid_thw = torch.cat(image_grid_thw, dim=0).to(device)
        targets = [{k: v.to(device) for k, v in t.items() if k != 'filename'} for t in targets]
        verb_losses, interaction_losses = model(samples, qwen_pixel_values, image_grid_thw, targets)

        losses = verb_losses + interaction_losses

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(verb_loss=verb_losses.item())
        metric_logger.update(interaction_loss=interaction_losses.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_hoi(dataset_file, model, postprocessors, data_loader,
                 subject_category_id, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    preds = []
    gts = []
    indices = []
    counter = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        # print(targets)
        outputs = model(samples, is_training=False)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['hoi'](outputs, orig_target_sizes)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(results))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))

        # counter += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    img_ids = [img_gts['id'] for img_gts in gts]
    _, indices = np.unique(img_ids, return_index=True)
    preds = [img_preds for i, img_preds in enumerate(preds) if i in indices]
    gts = [img_gts for i, img_gts in enumerate(gts) if i in indices]

    if dataset_file == 'hico':
        evaluator = HICOEvaluator(preds, gts, data_loader.dataset.rare_triplets,
                                  data_loader.dataset.non_rare_triplets, data_loader.dataset.correct_mat, args=args)
    elif dataset_file == 'vcoco':
        evaluator = VCOCOEvaluator(preds, gts, data_loader.dataset.correct_mat, use_nms_filter=args.use_nms_filter)

    stats = evaluator.evaluate()

    return stats
