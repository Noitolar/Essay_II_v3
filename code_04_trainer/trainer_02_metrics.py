import torch
import sklearn.metrics as skmetrics
import numpy as np


class MetricsCalculator:
    def __init__(self, num_labels: int):
        self.labels = list(range(num_labels))

    def topk_accuracy(
            self,
            preds: torch.Tensor | np.ndarray,  # 👈 <any>, <raw_x * raw_y>
            targets: torch.Tensor | np.ndarray,  # 👈 <any>, <raw_x * raw_y> or <any>
            topk: int
    ) -> float:
        assert type(preds) == type(targets)
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

        if len(targets.shape) == 1:
            return skmetrics.top_k_accuracy_score(y_score=preds, y_true=targets, k=topk, labels=self.labels)
        elif len(targets.shape) == 2:
            targets_label = targets.argmax(axis=-1)
            return skmetrics.top_k_accuracy_score(y_score=preds, y_true=targets_label, k=topk, labels=self.labels)
        else:
            raise NotImplementedError


if __name__ == "__main__":
    mtc = MetricsCalculator()

    num_samples = 4
    num_classes = 4

    p = torch.randn(num_samples, num_classes)
    p = torch.nn.functional.softmax(p, dim=-1)

    t = torch.eye(num_samples)

    acc_1 = mtc.topk_accuracy(p, t, topk=1)
    print(acc_1)
