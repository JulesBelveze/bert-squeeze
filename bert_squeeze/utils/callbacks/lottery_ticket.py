# This code is heavily based on:
# https://github.com/huggingface/transformers/blob/master/examples/research_projects/bertology/run_bertology.py

import logging
from datetime import datetime
from typing import Tuple

import lightning.pytorch as pl
import numpy as np
import torch
from pytorch_lightning.callbacks.base import Callback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm


class LotteryTicket(Callback):
    def __init__(
        self,
        normalize_by_layer: bool = False,
        normalize_global_importance: bool = False,
        masking_threshold: float = 0.9,
        masking_amount: float = 0.1,
        metric: str = "acc",
    ) -> None:
        """
        Args:
            normalize_by_layer (bool):
                normalize importance score by layers
            normalize_global_importance (bool):
                normalize all importance scores between 0 and 1
            masking_threshold (float):
                masking threshold in terms of metrics (stop masking when
                metric < threshold * original metric value)
            masking_amount (float):
                Amount of heads to mask at each masking step
        """
        super().__init__()
        assert (
            0.0 < masking_threshold < 1.0
        ), f"Masking threshold needs to be in [0.0, 1.0] got {masking_threshold}"

        self.normalize_by_layer = normalize_by_layer
        self.normalize_global_importance = normalize_global_importance
        self.masking_threshold = masking_threshold
        self.masking_amount = masking_amount

        self.metric = {
            "acc": accuracy_score,
            "f1": f1_score,
            "prec": precision_score,
            "rec": recall_score,
        }[metric]

    @staticmethod
    def entropy(p: torch.Tensor) -> float:
        """
        Computes the entropy of a probability distribution which represents
        the expected amount of information drawn from that distribution.

        Args:
            p (torch.Tensor):
                tensor of probabilities
        Returns:
            float:
                entropy value
        """
        plogp = p * torch.log(p)
        plogp[p == 0] = 0
        return -plogp.sum(dim=-1)

    @staticmethod
    def print_2d_tensor(tensor: torch.Tensor) -> None:
        """
        Print a 2D tensor

        Args:
            tensor (torch.Tensor):
                tensor to print
        """
        logging.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
        for row in range(len(tensor)):
            if tensor.dtype != torch.long:
                logging.info(
                    f"layer {row + 1}:\t"
                    + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data)
                )
            else:
                logging.info(
                    f"layer {row + 1}:\t"
                    + "\t".join(f"{x:d}" for x in tensor[row].cpu().data)
                )

    def compute_heads_importance(
        self,
        model: pl.LightningModule,
        eval_dataloader: DataLoader,
        compute_entropy: bool = True,
        compute_importance: bool = True,
        head_mask: torch.Tensor = None,
        actually_pruned: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.array]:
        """
        This method shows how to compute:
        - head attention entropy
        - head importance scores according to https://arxiv.org/abs/1905.10650

        Args:
            model (pl.LightningModule):
                lightning model for which to compute heads importance
            eval_dataloader (DataLoader):
                dataloader to use to compute the heads importance
            compute_entropy (bool):
                whether to compute the entropy of every attention layer
            compute_importance (bool):
                whether to compute the importance of every attention layer
            head_mask (torch.Tensor):
            actually_pruned (bool):

        Returns:
            torch.Tensor:
                computed entropy for every attention layer
            torch.Tensor:
                computed head importance of every head
            torch.Tensor:
                model's predictions
            np.array:
                ground truth labels
        """
        device = (
            model.device
        )  # since we are out of the `trainer` we need to specify the hardware

        # Prepare our tensors
        n_layers, n_heads = (
            model.model_config.num_hidden_layers,
            model.model_config.num_attention_heads,
        )
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

        for step, inputs in enumerate(tqdm(eval_dataloader, desc="Iteration")):
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
            outputs = model(**inputs, head_mask=head_mask, output_attentions=True)
            logits, all_attentions = outputs

            loss = model.loss(logits, inputs["labels"])
            loss.backward()  # Backpropagate to populate the gradients in the head mask

            if compute_entropy:
                for layer, attn in enumerate(all_attentions):
                    masked_entropy = self.entropy(attn.detach()) * inputs[
                        "attention_mask"
                    ].float().unsqueeze(1)
                    attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

            if compute_importance:
                head_importance += head_mask.grad.abs().detach()

            # Also store our logits/labels if we want to compute metrics afterwards
            if preds is None:
                preds = logits.detach().cpu().numpy()
                labels = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                labels = np.append(
                    labels, inputs["labels"].detach().cpu().numpy(), axis=0
                )

            tot_tokens += inputs["attention_mask"].float().detach().sum().data

        # Normalize
        attn_entropy /= tot_tokens
        head_importance /= tot_tokens

        # Layer wise importance normalization
        if not self.normalize_by_layer:
            exponent = 2
            norm_by_layer = torch.pow(
                torch.pow(head_importance, exponent).sum(-1), 1 / exponent
            )
            head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

        if not self.normalize_global_importance:
            head_importance = (head_importance - head_importance.min()) / (
                head_importance.max() - head_importance.min()
            )

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

    def mask_heads(
        self, model: pl.LightningModule, eval_dataloader: DataLoader
    ) -> torch.Tensor:
        """
        This method shows how to mask head (set some heads to zero), to test the effect on the network,
        based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)

        Args:
            model (pl.LightningModule):
                model to use
            eval_dataloader (DataLoader):
                dataloader to use to mask heads
        Returns:
            torch.Tensor:
                boolean mask for which heads to prune and not to prune
        """
        _, head_importance, preds, labels = self.compute_heads_importance(
            model, eval_dataloader, compute_entropy=False
        )
        preds = np.argmax(preds, axis=1)
        original_score = self.metric(preds, labels)
        logging.info(
            f"Pruning: original score: {original_score}, threshold: {original_score * self.masking_threshold}"
        )

        new_head_mask = torch.ones_like(head_importance)
        num_to_mask = max(1, int(new_head_mask.numel() * self.masking_amount))

        current_score = original_score
        while current_score >= original_score * self.masking_threshold:
            new_head_mask = new_head_mask.clone()
            head_mask = new_head_mask.clone()  # save current head mask
            # heads from least important to most - keep only not-masked heads
            head_importance[head_mask == 0.0] = float("Inf")
            current_heads_to_mask = head_importance.view(-1).sort()[1]

            if len(current_heads_to_mask) <= num_to_mask:
                break

            # mask heads
            current_heads_to_mask = current_heads_to_mask[:num_to_mask]
            logging.info(f"Heads to mask: {current_heads_to_mask.tolist()}")

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
            current_score = self.metric(preds, labels)
            logging.info(
                f"Masking: current score: {current_score}, remaining heads {new_head_mask.sum()} "
                f"({new_head_mask.sum() / new_head_mask.numel() * 100} percents)"
            )

        logging.info("Final head mask")
        self.print_2d_tensor(head_mask)
        np.save("head_mask.npy", head_mask.detach().cpu().numpy())

        return head_mask

    def prune_heads(
        self,
        model: pl.LightningModule,
        eval_dataloader: DataLoader,
        head_mask: torch.Tensor = None,
    ) -> None:
        """
        This method shows how to prune head (remove heads weights) based on the head
        importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)

        Args:
            model (pl.LightningModule):
                model to use
            eval_dataloader (DataLoader):
                dataloader to use to mask heads
            head_mask (torch.Tensor):
                boolean mask for which heads to prune and not to prune
        """
        # Try pruning and test time speedup
        # Pruning is like masking but we actually remove the masked weights
        before_time = datetime.now()
        _, _, preds, labels = self.compute_heads_importance(
            model,
            eval_dataloader,
            compute_entropy=False,
            compute_importance=False,
            head_mask=head_mask,
        )
        preds = np.argmax(preds, axis=1)
        score_masking = self.metric(preds, labels)
        original_time = datetime.now() - before_time

        original_num_params = sum(p.numel() for p in model.parameters())
        heads_to_prune = dict(
            (layer, (1 - head_mask[layer].long()).nonzero().squeeze().tolist())
            for layer in range(len(head_mask))
        )

        assert (
            sum(len(h) for h in heads_to_prune.values())
            == (1 - head_mask.long()).sum().item()
        )
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
        score_pruning = self.metric(preds, labels)
        new_time = datetime.now() - before_time

        logging.info(
            f"Pruning: original num of params: {original_num_params}, after pruning {pruned_num_params} "
            f"({pruned_num_params / original_num_params * 100} percents)"
        )
        logging.info(
            f"Pruning: score with masking: {score_masking} score with pruning: {score_pruning}"
        )
        logging.info(
            f"Pruning: speed ratio (new timing / original timing): {original_time / new_time * 100} percents"
        )

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Method called at the end of training to prune the model's heads.

        Args:
            trainer (pl.Trainer):
                lightning trainer
            pl_module (pl.LightningModule):
                lightning module
        """
        head_mask = self.mask_heads(
            model=pl_module, eval_dataloader=trainer.datamodule.val_dataloader()
        )
        self.prune_heads(
            model=pl_module,
            eval_dataloader=trainer.datamodule.val_dataloader(),
            head_mask=head_mask,
        )
