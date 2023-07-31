from transformers import Trainer
from typing import Dict

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