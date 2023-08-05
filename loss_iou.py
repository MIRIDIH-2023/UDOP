import torch

class IOU():
  def __init__(self):
    pass

  def iou(self,_input, _target):
    # Parse x1, y1, x2, y2 from location tokens
    
    iou_sum = 0
    num_step = 0

    for idx in range(0 , _input.size(0), 4):
        box1 = _input[idx:idx+4]
        box2 = _target[idx:idx+4]

        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])

        # 겹치는 영역의 너비와 높이 계산
        intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # 각 박스의 영역 계산
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # IOU 계산
        iou = intersection_area / (box1_area + box2_area - intersection_area)

        iou_sum+=iou
        num_step+=1

    return iou_sum/num_step
  
  def __call__(self, pred, target):
        # IOU 손실 함수 구현
        iou_loss = 1.0 - self.iou(pred, target)
        return iou_loss