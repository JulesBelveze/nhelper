from pathlib import Path
import os
from collections import defaultdict
from typing import List, Union, Callable, Optional

from .behavior import BehaviorSet, Behavior


class TestPack(object):
    """"""

    def __init__(self, behaviors: Optional[BehaviorSet] = None):
        self.behaviors = behaviors if behaviors is not None else BehaviorSet()
        self.outputs = []
        self._is_ran = False

    def add(self, new_behaviors: Union[Behavior, List[Behavior]]) -> None:
        """"""
        if not isinstance(new_behaviors, list):
            new_behaviors = [new_behaviors]
        self.behaviors.update(new_behaviors)

    def run(self) -> None:
        """"""
        if self._is_ran:
            raise ValueError("The 'TestPack' has already been ran.")
        self.outputs = [behavior.run() for behavior in self.behaviors]
        self._is_ran = True

    def get_performance(self):
        """"""
        if not self._is_ran:
            raise ValueError("You need to run the 'TestPack' to compute its performance.")

        performer_dict = defaultdict(list)
        for behavior in self.behaviors:
            performer_dict[behavior.task_type].append(behavior)

    def to_file(self, folder: str):
        """"""
        Path(folder).mkdir(parents=True, exist_ok=True)

        for behavior in self.behaviors:
            behavior.to_file(folder)

    @classmethod
    def from_file(cls, folder: str, prediction_fns: Union[List[Callable], Callable]):
        """"""
        files = [f for f in os.listdir(folder) if f.endswith("pkl")]

        if isinstance(prediction_fns, list):
            assert len(files) == len(prediction_fns), \
                "The number of prediction functions provided differs with the number of behaviors found"
        else:
            prediction_fns = [prediction_fns, ] * len(files)

        behaviors = BehaviorSet()
        for f, prediction_fn in zip(files, prediction_fns):
            behavior = Behavior.from_file(
                path_to_file=os.path.join(folder, f),
                predict_fn=prediction_fn
            )
            behaviors.add(behavior)

        return cls(behaviors)
