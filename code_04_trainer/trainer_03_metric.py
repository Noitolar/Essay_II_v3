import torch
import sklearn.metrics as skmetrics
import numpy as np


class MetricsCalculator:
    def __init__(self, num_classes: int):
        self.labels = list(range(num_classes))

    def topk_accuracy(
            self,
            preds: torch.Tensor | np.ndarray,  # 👈 <batch * seq>, <num_classes>
            targets: torch.Tensor | np.ndarray,  # 👈 <batch * seq>
            topk: int
    ) -> float:
        assert type(preds) == type(targets)
        if len(preds) == 0:
            return 0

        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

        acc = skmetrics.top_k_accuracy_score(y_score=preds, y_true=targets, k=topk, labels=self.labels)
        return acc


if __name__ == "__main__":

    classes = 1054
    mtc = MetricsCalculator(classes)
    p = torch.randn(1920, classes)
    t = torch.randint(low=0, high=classes, size=(1920,))

    acc_1 = mtc.topk_accuracy(p, t, topk=1)
    acc_5 = mtc.topk_accuracy(p, t, topk=5)
    acc_10 = mtc.topk_accuracy(p, t, topk=10)
    acc_20 = mtc.topk_accuracy(p, t, topk=20)
    print(acc_1)
    print(acc_5)
    print(acc_10)
    print(acc_20)
