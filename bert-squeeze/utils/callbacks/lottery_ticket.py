# This code is heavily based on:
# https://github.com/huggingface/transformers/blob/master/examples/research_projects/bertology/run_bertology.py

import logging
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.base import Callback
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm


class LotteryTicket(Callback):
    def __init__(self, masking_threshold: float = 0.9, masking_amount: float = 0.1, normalize_by_layer: bool = False,
                 normalize_global: bool = False):
        self.normalize_by_layer = normalize_by_layer
        self.normalize_global = normalize_global
        self.masking_threshold = masking_threshold
        self.masking_amount = masking_amount

    @staticmethod
    def entropy(p: torch.Tensor) -> float:
        """
        Computes the entropy of a probability distribution which represents
        the expected amount of information drawn from that distribution.
        """
        plogp = p * torch.log(p)
        plogp[p == 0] = 0
        return -plogp.sum(dim=-1)

    @staticmethod
    def print_2d_tensor(tensor: torch.Tensor) -> None:
        """Print a 2D tensor"""
        logging.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
        for row in range(len(tensor)):
            if tensor.dtype != torch.long:
                logging.info(f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data))
            else:
                logging.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))

    def compute_heads_importance(self, model, eval_dataloader: DataLoader, compute_entropy: bool = True,
                                 compute_importance: bool = True, head_mask: torch.Tensor = None,
                                 actually_pruned: bool = False):
        """This method shows how to compute:
        - head attention entropy
        - head importance scores according to http://arxiv.org/abs/1905.10650
        """
        device = model.device  # since we are out of the `trainer` we need to specify the hardware

        # Prepare our tensors
        n_layers, n_heads = model.model_config.num_hidden_layers, model.model_config.num_attention_heads
        head_importance = torch.zeros(n_layers, n_heads).to(device)
        attn_entropy = torch.zeros(n_layers, n_heads).to(device)

        if head_mask is None:
            head_mask = torch.ones(n_layers, n_heads).to(device)

        head_mask.requires_grad_(requires_grad=True)
        # If actually pruned attention multi-head, set head mask to None to avoid shape mismatch
        if actually_pruned:
            head_mask = None

        preds = None
        labels = None
        tot_tokens = 0.0

        for step, inputs in enumerate(
                tqdm(eval_dataloader, desc="Iteration")):  # , disable=args.local_rank not in [-1, 0])):

            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
            outputs = model(**inputs, head_mask=head_mask, output_attentions=True)
            logits, all_attentions = outputs

            loss = model.loss(logits, inputs["labels"])
            loss.backward()  # Backpropagate to populate the gradients in the head mask

            if compute_entropy:
                for layer, attn in enumerate(all_attentions):
                    masked_entropy = self.entropy(attn.detach()) * inputs["attention_mask"].float().unsqueeze(1)
                    attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

            if compute_importance:
                head_importance += head_mask.grad.abs().detach()

            # Also store our logits/labels if we want to compute metrics afterwards
            if preds is None:
                preds = logits.detach().cpu().numpy()
                labels = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)

            tot_tokens += inputs["attention_mask"].float().detach().sum().data

        # Normalize
        attn_entropy /= tot_tokens
        head_importance /= tot_tokens

        # Layerwise importance normalization
        if not self.normalize_by_layer:
            exponent = 2
            norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
            head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

        if not self.normalize_global:
            head_importance = (head_importance - head_importance.min()) / (
                    head_importance.max() - head_importance.min())

        # Print/save matrices
        np.save("attn_entropy.npy", attn_entropy.detach().cpu().numpy())
        np.save("head_importance.npy", head_importance.detach().cpu().numpy())

        logging.info("Attention entropies")
        self.print_2d_tensor(attn_entropy)
        logging.info("Head importance scores")
        self.print_2d_tensor(head_importance)
        logging.info("Head ranked by importance scores")
        head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=device)
        head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
            head_importance.numel(), device=device
        )
        head_ranks = head_ranks.view_as(head_importance)
        self.print_2d_tensor(head_ranks)

        return attn_entropy, head_importance, preds, labels

    def mask_heads(self, model, eval_dataloader):
        """This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
        """
        _, head_importance, preds, labels = self.compute_heads_importance(model, eval_dataloader, compute_entropy=False)
        preds = np.argmax(preds, axis=1)
        original_score = accuracy_score(preds, labels)
        logging.info("Pruning: original score: %f, threshold: %f", original_score,
                     original_score * self.masking_threshold)

        new_head_mask = torch.ones_like(head_importance)
        num_to_mask = max(1, int(new_head_mask.numel() * self.masking_amount))

        current_score = original_score
        while current_score >= original_score * self.masking_threshold:
            head_mask = new_head_mask.clone()  # save current head mask
            # heads from least important to most - keep only not-masked heads
            head_importance[head_mask == 0.0] = float("Inf")
            current_heads_to_mask = head_importance.view(-1).sort()[1]

            if len(current_heads_to_mask) <= num_to_mask:
                break

            # mask heads
            current_heads_to_mask = current_heads_to_mask[:num_to_mask]
            logging.info("Heads to mask: %s", str(current_heads_to_mask.tolist()))

            new_head_mask = new_head_mask.view(-1).detach()
            new_head_mask[current_heads_to_mask] = 0.0
            new_head_mask = new_head_mask.view_as(head_mask)
            new_head_mask = new_head_mask.clone()
            self.print_2d_tensor(new_head_mask)

            # Compute metric and head importance again
            _, head_importance, preds, labels = self.compute_heads_importance(
                model, eval_dataloader, compute_entropy=False, head_mask=new_head_mask
            )
            preds = np.argmax(preds, axis=1)
            current_score = accuracy_score(preds, labels)
            logging.info(
                "Masking: current score: %f, remaining heads %d (%.1f percents)",
                current_score,
                new_head_mask.sum(),
                new_head_mask.sum() / new_head_mask.numel() * 100,
            )

        logging.info("Final head mask")
        self.print_2d_tensor(head_mask)
        np.save("head_mask.npy", head_mask.detach().cpu().numpy())

        return head_mask

    def prune_heads(self, model, eval_dataloader, head_mask):
        """This method shows how to prune head (remove heads weights) based on
        the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
        """
        # Try pruning and test time speedup
        # Pruning is like masking but we actually remove the masked weights
        before_time = datetime.now()
        _, _, preds, labels = self.compute_heads_importance(
            model, eval_dataloader, compute_entropy=False, compute_importance=False, head_mask=head_mask
        )
        preds = np.argmax(preds, axis=1)
        score_masking = accuracy_score(preds, labels)
        original_time = datetime.now() - before_time

        original_num_params = sum(p.numel() for p in model.parameters())
        heads_to_prune = dict(
            (layer, (1 - head_mask[layer].long()).nonzero().squeeze().tolist()) for layer in range(len(head_mask))
        )

        assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()
        model.prune_heads(heads_to_prune)
        pruned_num_params = sum(p.numel() for p in model.parameters())

        before_time = datetime.now()
        _, _, preds, labels = self.compute_heads_importance(
            model,
            eval_dataloader,
            compute_entropy=False,
            compute_importance=False,
            head_mask=None,
            actually_pruned=True,
        )
        preds = np.argmax(preds, axis=1)
        score_pruning = accuracy_score(preds, labels)
        new_time = datetime.now() - before_time

        logging.info(
            "Pruning: original num of params: %.2e, after pruning %.2e (%.1f percents)",
            original_num_params,
            pruned_num_params,
            pruned_num_params / original_num_params * 100,
        )
        logging.info("Pruning: score with masking: %f score with pruning: %f", score_masking, score_pruning)
        logging.info("Pruning: speed ratio (new timing / original timing): %f percents", original_time / new_time * 100)

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """"""
        head_mask = self.mask_heads(model=pl_module, eval_dataloader=trainer.datamodule.val_dataloader())
        self.prune_heads(model=pl_module, eval_dataloader=trainer.datamodule.val_dataloader(), head_mask=head_mask)
