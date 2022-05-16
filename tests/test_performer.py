import pytest

from nlptest.behavior import SequenceClassificationBehavior
from nlptest.performers import SequenceClassificationPerformer
from nlptest.types import BehaviorType, TaskType


@pytest.fixture
def seq_classification_behavior():
    return SequenceClassificationBehavior(
        name="Test sequence classification",
        test_type=BehaviorType.invariance,
        task_type=TaskType.sequence_classification,
        samples=["This is a test"],
        labels=[1],
        predict_fn=lambda x: [1] * len(x)
    )


@pytest.fixture
def seq_classification_behavior2():
    return SequenceClassificationBehavior(
        name="Test sequence classification 2",
        test_type=BehaviorType.directional,
        task_type=TaskType.sequence_classification,
        samples=["This is a test", "This is a 2nd test"],
        labels=[2, 1],
        predict_fn=lambda x: [1] * len(x)
    )


class TestPerformer:
    """"""

    def test_metrics(self, seq_classification_behavior, seq_classification_behavior2):
        """"""
        performer = SequenceClassificationPerformer(
            behaviors=[seq_classification_behavior, seq_classification_behavior2],
            labels=[1, 2]
        )
        assert performer.result[BehaviorType.directional.value] == \
               performer.result["Test sequence classification 2"] == 0.5
        assert performer.result[BehaviorType.invariance.value] == \
               performer.result["Test sequence classification"] == 1.0
        assert performer.result["total"] == 2 / 3
