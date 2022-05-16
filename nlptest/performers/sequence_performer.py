import logging
from collections import defaultdict
from functools import reduce
from typing import Union, List
from tabulate import tabulate
import numpy as np

from nlptest.behavior import SequenceClassificationBehavior


class SequenceClassificationPerformer(object):
    """"""

    def __init__(self, behaviors: List[SequenceClassificationBehavior], labels: List[Union[int, str]],
                 metric_type: str = "weighted"):
        self.behaviors = behaviors
        self.labels = labels
        self.metric_type = metric_type

        self.eps = 1e-8

        self._fit(behaviors)

    def _fit(self, behaviors: List[SequenceClassificationBehavior]) -> None:
        """"""
        if not behaviors[0]._is_ran:
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

    def tabulate_result(self):
        """"""
        return tabulate({key: list(value) for key, value in self.result.items()}, headers="keys")
