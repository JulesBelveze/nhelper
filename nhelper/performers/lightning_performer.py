from collections import defaultdict
from functools import reduce
from typing import Any, Callable, Optional

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class LightningPerformer(pl.Callback):
    def __init__(self, postprocessor: Optional[Callable] = None):
        super(LightningPerformer, self).__init__()
        self.postprocessor = postprocessor

        self.result = None
        self.all = []
        self.per_capability = defaultdict(list)
        self.per_name = defaultdict(list)
        self.per_type = defaultdict(list)

    def on_test_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: Optional[STEP_OUTPUT],
                          batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self.postprocessor:
            success = self.postprocessor(batch, outputs)
        else:
            success = batch["labels"] == outputs

        self.all.append(success)
        self.per_capability[batch["capability"]].append(success)
        self.per_name[batch["name"]].append(success)
        self.per_type[batch["test_type"]].append(success)

    def on_test_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        total_success = {"Total": [np.mean(self.all), f"{np.sum(self.all)}/{len(self.all)}"]}
        per_name_success = {
            f"Name - {key}": [np.mean(val), f"{np.sum(val)}/{len(val)}"] for key, val in self.per_name.items()
        }
        per_capability_success = {
            f"Capability - {key}": [np.mean(val), f"{np.sum(val)}/{len(val)}"] for key, val in
            self.per_capability.items()
        }
        per_type_success = {
            f"Behavior type - {key}": [np.mean(val), f"{np.sum(val)}/{len(val)}"] for key, val in self.per_type.items()
        }
        self.result = reduce(lambda x, y: dict(x, **y),
                             (total_success, per_name_success, per_capability_success, per_type_success))

        for logger in trainer.loggers:
            logger.log_hyperparams(self.result)
