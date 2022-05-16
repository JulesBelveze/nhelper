import pytest

from nlptest.behavior import SequenceClassificationBehavior, DuplicateBehaviorError
from nlptest.testpack import TestPack
from nlptest.types import BehaviorType, TaskType


@pytest.fixture
def seq_classification_behavior():
    return SequenceClassificationBehavior(
        name="Test sequence classification",
        test_type=BehaviorType.invariance,
        task_type=TaskType.sequence_classification,
        samples=["TEST"],
        labels=[1],
        predict_fn=lambda x: 1
    )


@pytest.fixture
def seq_classification_behavior2():
    return SequenceClassificationBehavior(
        name="Test sequence classification 2",
        test_type=BehaviorType.invariance,
        task_type=TaskType.sequence_classification,
        samples=["TEST"],
        labels=[2],
        predict_fn=lambda x: 1
    )


class TestTestPack:
    """"""

    def test_add_behavior(self, seq_classification_behavior):
        """"""
        testpack = TestPack()
        testpack.add(seq_classification_behavior)

        assert len(testpack.behaviors) == 1

        with pytest.raises(DuplicateBehaviorError):
            testpack.add(seq_classification_behavior)

        assert len(testpack.behaviors) == 1

    def test_run(self, seq_classification_behavior, seq_classification_behavior2):
        """"""
        testpack = TestPack()
        testpack.add([seq_classification_behavior, seq_classification_behavior2])

        testpack.run()
        with pytest.raises(ValueError):
            testpack.run()