from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class CurriculumTrainer(Trainer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.save_epoch = -1
    self.ratio = 0.75

  def compute_new_ratio(self, epoch):
    self.save_epoch = epoch
    
    if epoch <= 10 :
      return 0.1
    elif epoch <= 15 :
      return 0.3
    elif epoch <= 20 :
      return 0.5
    elif epoch <= 25 :
      return 0.75
    else:
      return 1.0

  def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
    if self.args.save_total_limit == 1 and self.save_epoch == (epoch-1):

      self.ratio = self.compute_new_ratio(epoch) 
      
      self.train_dataset.set_layout_modeling_masking_ratio(self.ratio)
      
    super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
      
  def log(self, logs: Dict[str, float]) -> None:
    """
    Log `logs` on the various objects watching training.

    Subclass and override this method to inject custom behavior.

    Args:
        logs (`Dict[str, float]`):
            The values to log.
    """
    if self.state.epoch is not None:
      logs["epoch"] = round(self.state.epoch, 2)
        
    if self.ratio is not None :
      logs["masking ratio"] = self.ratio

    output = {**logs, **{"step": self.state.global_step}}
    self.state.log_history.append(output)
    self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

  def compute_loss(self, model, inputs, return_outputs=False):
      labels = inputs.get("labels")
      logits = model(**inputs).logits
      max_logits_indices = torch.argmax(logits, dim=2)

      weight = torch.ones(logits.size(2))  # Initialize with weight 1 for all classes
      weight[32500:33001] = 0  # Set weight 0 for the specified range [33000, 32500]

      loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)

      # Compute Cross Entropy (CE) loss for all elements in the logits and labels tensor.
      ce_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

      # Mask for elements if <loc_{idx}> in range [33000, 32500] inclusive
      mask_mse = ((labels >= 32500) & (labels <= 33000)) & ((max_logits_indices >= 32500) & (max_logits_indices <= 33000))

      huber_loss = F.smooth_l1_loss(max_logits_indices[mask_mse].float(), labels[mask_mse].float())

      loss = ce_loss + huber_loss

      return (loss, logits) if return_outputs else loss

