from transformers import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import IntervalStrategy

from typing import Dict, Optional

import numpy as np
import logging

logger = logging.getLogger(__name__)

class CurriculumTrainer(Trainer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.lm_ratio = None
      
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
      
    self.lm_ratio = self.train_dataset.dataset.get_layout_modeling_masking_ratio()
    if self.lm_ratio is not None :
      logs["masking ratio"] = self.lm_ratio

    output = {**logs, **{"step": self.state.global_step}}
    self.state.log_history.append(output)
    self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
    
class elevateMRCallback(TrainerCallback):
  
  def __init__(self, train_dataset, eval_dataset, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
    self.train_d = train_dataset
    self.eval_d = eval_dataset
    self.min_loss_per_MR = None # arbitrary MAX loss value
    self.early_stopping_patience = early_stopping_patience
    self.early_stopping_threshold = early_stopping_threshold
    # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
    self.early_stopping_patience_counter = 0
        
  def check_metric_value(self, args, state, control, metric_value):
    # best_metric is set by code for load_best_model
    operator = np.less
    if self.min_loss_per_MR is None or (
        operator(metric_value, self.min_loss_per_MR)
        and abs(metric_value - self.min_loss_per_MR) > self.early_stopping_threshold
    ):
      self.early_stopping_patience_counter = 0
      self.min_loss_per_MR = metric_value
    else:
      self.early_stopping_patience_counter += 1

  def on_train_begin(self, args, state, control, **kwargs):
    assert (
      args.evaluation_strategy != IntervalStrategy.NO
    ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

  def on_evaluate(self, args, state, control, metrics, **kwargs):
    metric_to_check = "loss"
    if not metric_to_check.startswith("eval_"):
      metric_to_check = f"eval_{metric_to_check}"
    metric_value = metrics.get(metric_to_check)

    if metric_value is None:
      logger.warning(
          f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
          " is disabled"
      )
      return

    self.check_metric_value(args, state, control, metric_value)
    if self.early_stopping_patience_counter >= self.early_stopping_patience:
      lm_ratio = self.eval_d.dataset.get_layout_modeling_masking_ratio()
      if lm_ratio == 1.0 :
        control.should_training_stop = True
      
      # elevate layout modeling masking ratio
      if lm_ratio != 1.0:
        lm_ratio += 0.05
        lm_ratio = round(lm_ratio, 2)
        self.train_d.dataset.set_layout_modeling_masking_ratio(lm_ratio)
        self.eval_d.dataset.set_layout_modeling_masking_ratio(lm_ratio)
      
      # initialize min_loss_per_MR. New masking ratio, new min_loss.
      self.min_loss_per_MR = None