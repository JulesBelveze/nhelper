from collections import defaultdict
from typing import List, Union

from .behavior import BehaviorSet, Behavior


class TestPack:
    """"""

    def __init__(self):
        self.behaviors = BehaviorSet()
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
