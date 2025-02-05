import numpy as np
import torch


class TrainerCache:
    def __init__(self) -> None:
        self.preds_cache = None
        self.targets_cache = None
        self.loss_cache = 0.0
        self.num_samples_cache = 0

    def update_cache(
            self,
            preds: torch.Tensor,
            targets: torch.Tensor,
            loss: torch.Tensor
    ) -> None:
        assert preds.shape == targets.shape
        num_samples = preds.shape[0]

        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        loss = float(loss.item())

        self.loss_cache += num_samples * loss
        self.num_samples_cache += num_samples

        if self.preds_cache is None:
            assert self.targets_cache is None
            self.preds_cache = preds
            self.targets_cache = targets
        else:
            assert self.targets_cache is not None
            self.preds_cache = np.append(self.preds_cache, preds, axis=0)
            self.targets_cache = np.append(self.targets_cache, targets, axis=0)

    def clear_cache(self) -> None:
        self.preds_cache = None
        self.targets_cache = None
        self.loss_cache = 0.0
        self.num_samples_cache = 0








