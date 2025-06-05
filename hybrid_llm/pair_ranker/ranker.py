# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from typing import Dict, List
import logging

# set up logging basic config from info
logging.basicConfig(level=logging.INFO)


class NClassRerankerLossType(Enum):
    DET_2CLS = "det_2cls"
    PROB_2CLS = "prob_2cls"
    DET_NCLS = "det_ncls"
    DET_NLABELS = "det_nlabels"
    PROB_NLABELS = "prob_nlabels"


class NClassReranker(nn.Module):

    def __init__(self, pretrained_model, args, tokenizer):
        """
        Initializes the NClassReranker class.

        Args:
            pretrained_model (PretrainedModel): The pretrained model used for reranking.
            args (Namespace): The arguments passed to the class.
            tokenizer (Tokenizer): The tokenizer used for tokenization.
        """
        super(NClassReranker, self).__init__()
        self.args = args
        self.n_tasks = self.args.n_tasks
        self.num_pos = self.args.num_pos
        self.num_neg = self.args.num_neg
        self.n_candidates = self.args.n_candidates
        self.t = self.args.match_t
        self.sub_sampling_mode = self.args.sub_sampling_mode
        self.sub_sampling_ratio = self.args.sub_sampling_ratio
        self.loss_type = self.args.loss_type
        self.drop_out = self.args.drop_out
        self.inference_mode = self.args.inference_mode
        if hasattr(pretrained_model.config, "is_encoder_decoder"):
            self.is_encoder_decoder = pretrained_model.config.is_encoder_decoder
        else:
            self.is_encoder_decoder = False

        self.pretrained_model = pretrained_model
        self.hidden_size = pretrained_model.config.out_hidden_state_size
        self.sep_token_id = (
            tokenizer.sep_token_id
            if tokenizer.sep_token_id is not None
            else tokenizer.eos_token_id
        )
        self.tokenizer = tokenizer

        self.head_layer = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(1 * self.hidden_size, self.n_candidates),
        )

        self.multilabel_loss = nn.BCEWithLogitsLoss()

        self.single_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(self.drop_out),
                    nn.Linear(1 * self.hidden_size, 2),
                )
                for _ in range(self.n_candidates - 1)
            ]
        )
        self.sigmoid = nn.Sigmoid()

    def _match_prob(
        self, score_small: List[float], score_large: List[float], t: float = 0.0
    ):
        """
        Calculates the match probability between two lists of scores.

        Args:
            score_small (List[float]): The list of scores for the smaller group.
            score_large (List[float]): The list of scores for the larger group.
            t (float, optional): The threshold value for considering a match. Defaults to 0.0.

        Returns:
            float: The match probability between the two groups.
        """
        len_s = len(score_small)
        len_l = len(score_large)

        return sum([sum(i >= score_large - t) / len_l for i in score_small]) / len_s

    def _calculate_loss(self, device, pred_labels: torch.Tensor, scores: torch.Tensor, costs: torch.Tensor):
        """
        Calculates the loss for the given predicted labels and scores.

        Args:
            device (torch.device): The device on which the tensors are located.
            pred_labels (torch.Tensor): The predicted labels.
            scores (torch.Tensor): The scores.

        Returns:
            torch.Tensor: The calculated loss.

        Raises:
            ValueError: If the loss type is unknown.
        """

        loss = torch.tensor(0.0, device=device)

        # logging scores shape
        # logging.info(scores.shape)

        if self.loss_type.startswith("det") and scores.dim() == 4:
            scores = scores[:, :, :, 0]

        if self.loss_type == NClassRerankerLossType.DET_2CLS.value:
            true_labels = torch.tensor(
                [0 if _[0] >= _[1] - self.t else 1 for _ in scores.squeeze(2).cpu()],
                device=device,
            )
            loss += F.cross_entropy(pred_labels, true_labels)
        elif self.loss_type == NClassRerankerLossType.PROB_2CLS.value:
            assert scores.dim() == 4
            new_scores = scores.squeeze(2).cpu()
            match_prob = [self._match_prob(s[0], s[1], t=self.t) for s in new_scores]
            match_prob = torch.tensor([[p, 1.0 - p] for p in match_prob], device=device)

            loss += F.cross_entropy(pred_labels, match_prob)
        elif self.loss_type == NClassRerankerLossType.DET_NCLS.value:
            # logging.info(scores.shape)
            true_labels = torch.argmax(scores.squeeze(2).cpu(), dim=1).to(device)
            loss += F.cross_entropy(pred_labels, true_labels)
        elif self.loss_type == NClassRerankerLossType.DET_NLABELS.value:
            true_labels = torch.tensor(
                [ [(_ >= s[-1]) * 1. for _ in s] for s in scores.squeeze(2).cpu()],
                device=device,
            )
            loss += self.multilabel_loss(pred_labels, true_labels)
        elif self.loss_type == NClassRerankerLossType.PROB_NLABELS.value:
            assert scores.dim() == 4
            new_scores = scores.squeeze(2).cpu()
            for i, pred in enumerate(pred_labels):
                match_prob = [self._match_prob(s[i], s[-1], t=self.t) for s in new_scores]
                match_prob = torch.tensor([[p, 1.0 - p] for p in match_prob], device=device)

                loss += F.cross_entropy(pred, match_prob)
            loss /= len(pred_labels)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    def _forward(
        self,
        source_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        scores: torch.Tensor,
        costs: torch.Tensor,
    ) -> Dict:
        """
        Forward pass of the ranker model.

        Args:
            source_ids (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing the input source IDs.
            source_attention_mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) containing the attention mask for the input source IDs.
            scores (torch.Tensor): Tensor of shape (batch_size, num_choices, num_choices, 1) containing the scores.

        Returns:
            dict: A dictionary containing the following keys:
                - "loss" (torch.Tensor): The computed loss value.
                - "preds" (torch.Tensor): The predicted labels.

        Raises:
            ValueError: If the loss type is unknown.
        """
        device = source_ids.device

        outputs = self.pretrained_model(
            input_ids=source_ids,
            attention_mask=source_attention_mask,
            output_hidden_states=True,
        )
        encs = outputs.hidden_states[-1]
        source_idxs = torch.where(source_ids == self.tokenizer.source_prefix_id)
        source_encs = encs[source_idxs[0], source_idxs[1], :]
        if self.loss_type == NClassRerankerLossType.PROB_NLABELS.value:
            pred_labels = []
            for head in self.single_heads:
                pred = head(source_encs)
                pred_labels.append(pred)
        else:
            pred_labels = self.head_layer(source_encs)
        loss = torch.tensor(0.0, device=device)

        if self.pretrained_model.training:
            loss = self._calculate_loss(device, pred_labels, scores, costs)

        return {
            "loss": loss,
            "preds": pred_labels,
        }

    def predict(
        self,
        source_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        scores: torch.Tensor = None,
        costs: torch.Tensor = None,
    ) -> Dict:
        """
        Predicts the output using the given inputs.

        Args:
            source_ids (torch.Tensor): The input source IDs.
            source_attention_mask (torch.Tensor): The attention mask for the source IDs.
            scores (torch.Tensor, optional): The scores for the predictions. Defaults to None.

        Returns:
            dict: A dictionary containing the following keys:
                - "loss" (torch.Tensor): The computed loss value.
                - "preds" (torch.Tensor): The predicted labels.

        """
        return self._forward(source_ids, source_attention_mask, scores, costs)

    def forward(
        self,
        source_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        scores: torch.Tensor,
        costs: torch.Tensor,
    ) -> Dict:
        """
        Performs the forward pass of the ranker model.

        Args:
            source_ids (torch.Tensor): The input source IDs.
            source_attention_mask (torch.Tensor): The attention mask for the source IDs.
            scores (torch.Tensor): The input scores.

        Returns:
            dict: A dictionary containing the following keys:
                - "loss" (torch.Tensor): The computed loss value.
                - "preds" (torch.Tensor): The predicted labels.
        """
        return self._forward(source_ids, source_attention_mask, scores, costs)
