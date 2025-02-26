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
            loss: torch.Tensor,
            marked_matrix: torch.Tensor,
    ) -> None:
        assert len(preds.shape) == 2
        assert len(targets.shape) == 1
        assert len(marked_matrix.shape) == 1
        assert preds.shape[0] == targets.shape[0] == marked_matrix.shape[0], f"{preds.shape}, {targets.shape}, {marked_matrix.shape}"

        n = preds.shape[0]

        preds = preds.detach().cpu()
        targets = targets.detach().cpu()
        marked_matrix = marked_matrix.detach().cpu()

        marked_indexes = torch.where(marked_matrix == True)[0]
        marked_preds = torch.index_select(preds, dim=0, index=marked_indexes)
        marked_targets = torch.index_select(targets, dim=0, index=marked_indexes)

        marked_preds = marked_preds.numpy()
        marked_targets = marked_targets.numpy()
        loss = float(loss.item())

        self.loss_cache += n * loss
        self.num_samples_cache += n

        if self.preds_cache is None:
            assert self.targets_cache is None
            self.preds_cache = marked_preds
            self.targets_cache = marked_targets
        else:
            assert self.targets_cache is not None
            self.preds_cache = np.append(self.preds_cache, marked_preds, axis=0)
            self.targets_cache = np.append(self.targets_cache, marked_targets, axis=0)

    def clear_cache(self) -> None:
        self.preds_cache = None
        self.targets_cache = None
        self.loss_cache = 0.0
        self.num_samples_cache = 0








