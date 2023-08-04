import torch
import tqdm as tqdm

from ..common.utils import calculate_iou


def compute_custom_metrics(self, model, dataset=None):
    metrics = {
        "accuracy": 0,
        "mae": 0,
        "iou": 0
    }

    correct_predictions = 0
    total_predictions = 0
    mae_sum = 0
    mae_count = 0
    iou_sum = 0
    iou_count = 0

    prev_mask_ratio = dataset.dataset.get_layout_modeling_masking_ratio()
    dataset.dataset.set_layout_modeling_masking_ratio(1.0)

    for i in tqdm(range(len(dataset))):
        sample = self.data_collator([dataset[i]])

        # Move tensors to the model's device
        for key, value in sample.items():
            if torch.is_tensor(value):
                sample[key] = value.to(self.model.device)

        with torch.no_grad():
            logits = model(**sample).logits
            pred = torch.argmax(logits, dim=2)
            label = sample['labels'].to(self.model.device)

        # Find the index of the last true label (where label equals 1)
        last_true_label_index = (label == 1).nonzero(as_tuple=False).max()

        # Slice the pred and label tensors up to the last true label index
        pred_sliced = pred[:, :last_true_label_index + 1]
        label_sliced = label[:, :last_true_label_index + 1]

        # Calculate accuracy for this batch
        correct_predictions += torch.sum(pred_sliced == label_sliced).item()
        total_predictions += pred_sliced.numel()

        # Apply mask for <loc> tokens
        mask_mse = ((label_sliced >= 32500) & (label_sliced <= 33000)) & ((pred_sliced >= 32500) & (pred_sliced <= 33000))

        if torch.any(mask_mse):
            mae_sum += torch.abs(label_sliced[mask_mse] - pred_sliced[mask_mse]).sum().item()
            mae_count += mask_mse.sum().item()


        # Calculate IOU for <loc> tokens
        for idx in range(mask_mse.size(1) - 3):
            if torch.all(mask_mse[0, idx:idx + 4]):  # Check for four consecutive True values
                pred_box = [self.tokenizer.decode(token_id) for token_id in pred_sliced[0, idx:idx + 4]] # Extract 4 tokens for the bounding box
                label_box = [self.tokenizer.decode(token_id) for token_id in label_sliced[0, idx:idx + 4]]  

                iou_sum += calculate_iou(pred_box, label_box)
                iou_count += 1

    # Calculate metrics
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    mae = mae_sum / mae_count if mae_count > 0 else 0
    iou = iou_sum / iou_count if iou_count > 0 else 0

    metrics["accuracy"] = accuracy
    metrics["mae"] = mae
    metrics["iou"] = iou

    dataset.dataset.set_layout_modeling_masking_ratio(prev_mask_ratio)

    return metrics

