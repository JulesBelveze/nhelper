import logging
from collections import defaultdict
from functools import reduce
from typing import Union, List

import numpy as np
from tabulate import tabulate

from nlptest.behavior import Behavior


class Performer(object):
    """Object use to compute a performance summary of a list of Behaviors."""

    def __init__(self, metric_type: str = "weighted"):
        """
        :param metric_type: aggregation type
        """
        self.metric_type = metric_type

        self.eps = 1e-8
        self._is_fitted = False

    def fit(self, behaviors: List[Behavior]) -> None:
        """"""
        if self._is_fitted:
            raise ValueError("Performer is already fitted.")

        if not all([behavior._is_ran for behavior in behaviors]):
            logging.info(f"The behaviors were not run, running them now...")
            [b.run() for b in behaviors]

        flatten_outputs = [output for behavior in behaviors for output in behavior.outputs]
        total_acc = {"total": np.mean([out.success for out in flatten_outputs])}

        # Retrieving accuracy per 'Behavior'
        per_name_acc = {
            behavior.name: np.mean([output.success for output in behavior.outputs]) for behavior in behaviors
        }

        # Retrieving accuracy per 'BehaviorType'
        per_behavior_type_acc = defaultdict(list)
        for behavior in behaviors:
            per_behavior_type_acc[behavior.test_type.value].extend([output.success for output in behavior.outputs])

        per_behavior_type_acc = {key: np.mean(value) for key, value in per_behavior_type_acc.items()}

        self.result = reduce(lambda x, y: dict(x, **y), (total_acc, per_name_acc, per_behavior_type_acc))
        logging.info("'Performer' has been successfully fitted.")
        self._is_fitted = True

    def tabulate_result(self):
        """Prettify results"""
        return tabulate([[key, value] for key, value in self.result.items()])
