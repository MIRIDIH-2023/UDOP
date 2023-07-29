from transformers import Trainer

class CurriculumTrainer(Trainer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.save_epoch = -1

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
    super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

    if self.args.curriculm == 'yes' and self.save_epoch == (epoch-1):
      new_ratio = 0.75
      new_ratio = self.compute_new_ratio(epoch) 
      
      self.train_dataset.set_layout_modeling_masking_ratio(new_ratio)