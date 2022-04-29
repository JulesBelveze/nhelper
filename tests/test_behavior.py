import random
from typing import List

import pytest

from nlptest.behavior import SequenceClassificationBehavior, SpanClassificationBehavior
from nlptest.types import BehaviorType, TaskType, Span


@pytest.fixture
def seq_classification_behavior():
    return SequenceClassificationBehavior(test_type=BehaviorType.invariance, task_type=TaskType.sequence_classification)


@pytest.fixture
def span_classification_behavior():
    return SpanClassificationBehavior(test_type=BehaviorType.invariance, task_type=TaskType.span_classification)


@pytest.fixture
def random_class():
    return random.randint(0, 10)


@pytest.fixture
def random_span():
    return Span(start=random.randint(0, 100), end=random.randint(100, 200), label=random.randint(0, 4))


@pytest.fixture
def text_sample():
    return "My name is Wolfgang and I live in Berlin"


class TestSequenceClassificationBehavior:
    """"""

    @staticmethod
    def predict_fn(list_text: List[str]):
        return [random.randint(0, 10)] * len(list_text)

    def test_run(self, text_sample, random_class):
        n_samples = 5
        behavior = SequenceClassificationBehavior(
            test_type=BehaviorType.invariance,
            task_type=TaskType.sequence_classification,
            samples=[text_sample] * n_samples,
            labels=[random_class] * n_samples,
            predict_fn=self.predict_fn
        ).run()
        assert len(behavior) == n_samples
        assert all([b.y_pred is not None for b in behavior])


class TestSpanClassificationBehavior:
    """"""

    @staticmethod
    def predict_fn(list_text: List[str]):
        return [[Span(start=0, end=10, label=1), Span(start=20, end=30, label=1)],] * len(list_text)

    def test_run(self, text_sample, random_span):
        n_samples = 5
        behavior = SpanClassificationBehavior(
            test_type=BehaviorType.invariance,
            task_type=TaskType.sequence_classification,
            samples=[text_sample] * n_samples,
            labels=[[random_span,] * 4] * n_samples,
            predict_fn=self.predict_fn
        ).run()
        assert len(behavior) == n_samples
        assert all([b.y_pred is not None for b in behavior])
