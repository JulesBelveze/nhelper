import logging
from collections import defaultdict
from functools import reduce
from typing import List

import numpy as np
from tabulate import tabulate

from nhelper.behavior import Behavior


class Performer(object):
    """Object use to compute a performance summary of a list of Behaviors."""

    def __init__(self, metric_type: str = "weighted", binarize: bool = False):
        """
        :param metric_type: aggregation type
        :param binarize: whether to compute performance on binarized predictions.
        """
        self.metric_type = metric_type
        self.success_attr = "success" if not binarize else "binary_success"
        print(self.success_attr)

        self.eps = 1e-8
        self._is_fitted = False
        self.result = None

    def fit(self, behaviors: List[Behavior]) -> None:
        """
        :param behaviors: list of Behavior to test on
        :return:
        """
        if self._is_fitted:
            raise ValueError("Performer is already fitted.")

        if not all([behavior._is_ran for behavior in behaviors]):
            logging.info(f"The behaviors were not run, running them now...")
            [b.run() for b in behaviors]

        flatten_outputs = [output for behavior in behaviors for output in behavior.outputs]

        total_success = [getattr(output, self.success_attr) for output in flatten_outputs]
        total_acc = {"Total": [np.mean(total_success), f"{np.sum(total_success)}/{len(total_success)}"]}

        # Retrieving accuracy per 'Behavior'
        named_success = {
            f"Name - {behavior.name}": [getattr(output, self.success_attr) for output in behavior.outputs]
            for behavior in behaviors
        }

        per_name_acc = {
            key: [np.mean(val), f"{np.sum(val)}/{len(val)}"] for key, val in named_success.items()
        }

        # Retrieving accuracy per 'Behavior.capability'
        per_capability_acc = defaultdict(list)
        for behavior in behaviors:
            per_capability_acc[f"Capability - {behavior.capability}"].extend(
                [getattr(output, self.success_attr) for output in behavior.outputs]
            )
        per_capability_acc = {
            key: [np.mean(val), f"{np.sum(val)}/{len(val)}"] for key, val in per_capability_acc.items()
        }

        # Retrieving accuracy per 'BehaviorType'
        per_behavior_type_acc = defaultdict(list)
        for behavior in behaviors:
            per_behavior_type_acc[f"Behavior type - {behavior.test_type.value}"].extend(
                [getattr(output, self.success_attr) for output in behavior.outputs]
            )
        per_behavior_type_acc = {
            key: [np.mean(val), f"{np.sum(val)}/{len(val)}"] for key, val in per_behavior_type_acc.items()
        }

        self.result = reduce(lambda x, y: dict(x, **y),
                             (total_acc, per_name_acc, per_behavior_type_acc, per_capability_acc))
        logging.info("'Performer' has been successfully fitted.")
        self._is_fitted = True

    def tabulate_result(self):
        """Prettify results"""
        return tabulate([[key] + value for key, value in self.result.items()], headers=["Test", "Acc", "Support"])
