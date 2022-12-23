import os
from pathlib import Path
from typing import Callable, List, Optional, Union

from nhelper.behavior import Behavior, BehaviorSet
from nhelper.performers import PerformerType


class TestPack(object):
    """A 'TestPack' is intended to centralize the different 'Behaviors' of a test suite."""

    def __init__(self, behaviors: Optional[BehaviorSet] = None, performer: PerformerType = None):
        """

        :param behaviors: a set of Behaviors to be added to the test suite
        :param performer: object to use to compute performance summary
        """
        self.behaviors = behaviors if behaviors is not None else BehaviorSet()
        self.performer = performer
        self.outputs = []
        self._is_ran = False

    @property
    def result(self):
        if not self._is_ran:
            return None
        return self.performer.result

    def add(self, new_behaviors: Union[Behavior, List[Behavior]]) -> None:
        """
        Adds new Behavior(s) to the current test suite

        :param new_behaviors: behavior(s) to add to the test suite
        :return:
        """
        if not isinstance(new_behaviors, list):
            new_behaviors = [new_behaviors]
        self.behaviors.update(new_behaviors)

    def run(self) -> None:
        """
        Runs the different Behaviors
        """
        if self._is_ran:
            raise ValueError("The 'TestPack' has already been ran.")
        self.outputs = [behavior.run() for behavior in self.behaviors]

        self.performer.fit(self.behaviors)
        self._is_ran = True

    def to_file(self, folder: str):
        """
        Saves the current Behaviors contained in the pack into pickle objects.

        :param folder: path to save the current TestPack
        :return:
        """
        Path(folder).mkdir(parents=True, exist_ok=True)

        for behavior in self.behaviors:
            behavior.to_file(folder)

    @classmethod
    def from_file(cls, folder: str, prediction_fns: Union[List[Callable], Callable] = None,
                  performer: PerformerType = None):
        """
        Loads a TestPack from a folder

        :param folder: path to the saved TestPack folder
        :param prediction_fns: function(s) to use to obtain predictions. Either pass a single function
                               that will be used for all Behaviors or one per Behavior.
        :param performer: object to compute performance summary
        :return: TestPack
        """
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

        return cls(behaviors, performer)
