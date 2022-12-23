import random
from typing import List

import pytest

from nhelper.behavior import MultiLabelSequenceClassificationBehavior, SequenceClassificationBehavior, \
    SpanClassificationBehavior, TokenClassificationBehavior
from nhelper.types import BehaviorType, Span, Token


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
            capability="Capability 1",
            name="Test sequence classification",
            test_type=BehaviorType.invariance,
            samples=[text_sample] * n_samples,
            labels=[random_class] * n_samples,
            predict_fn=self.predict_fn
        )
        behavior.run()
        assert len(behavior.outputs) == n_samples
        assert all([b.y_pred is not None for b in behavior.outputs])

        with pytest.raises(ValueError):
            behavior.run()

    def test_save_and_load(self, text_sample, random_class):
        """"""
        n_samples = 5
        behavior = SequenceClassificationBehavior(
            capability="Capability 2",
            name="Test sequence classification",
            test_type=BehaviorType.invariance,
            samples=[text_sample] * n_samples,
            labels=[random_class] * n_samples,
            predict_fn=self.predict_fn
        )
        behavior.run()
        behavior.to_file("tmp_data/")
        output0 = behavior.outputs

        new_behavior = SequenceClassificationBehavior.from_file(
            "tmp_data/Test_sequence_classification.pkl",
            self.predict_fn
        )
        new_behavior.run()
        output1 = behavior.outputs

        assert output0 == output1


class TestMultiLabelSequenceClassificationBehavior:
    """"""
    n_labels = 4

    def predict_fn(self, list_text: List[str]):
        labels = [random.randint(0, 1), ] * self.n_labels
        return [labels, ] * len(list_text)

    def test_run(self, text_sample):
        n_samples = 5

        behavior = MultiLabelSequenceClassificationBehavior(
            capability="Capability 1",
            name="Test multi label sequence classification",
            test_type=BehaviorType.invariance,
            samples=[text_sample] * n_samples,
            labels=[[1, ] * self.n_labels] * n_samples,
            predict_fn=self.predict_fn
        )
        behavior.run()
        assert len(behavior.outputs) == n_samples
        assert all([b.y_pred is not None for b in behavior.outputs])

        with pytest.raises(ValueError):
            behavior.run()

    def test_save_and_load(self, text_sample):
        """"""
        n_samples = 5

        behavior = MultiLabelSequenceClassificationBehavior(
            capability="Capability 2",
            name="Test multi label sequence classification",
            test_type=BehaviorType.invariance,
            samples=[text_sample] * n_samples,
            labels=[[1, ] * self.n_labels] * n_samples,
            predict_fn=self.predict_fn
        )
        behavior.run()
        behavior.to_file("tmp_data/")
        output0 = behavior.outputs

        new_behavior = MultiLabelSequenceClassificationBehavior.from_file(
            "tmp_data/Test_multi_label_sequence_classification.pkl",
            self.predict_fn
        )
        new_behavior.run()
        output1 = behavior.outputs

        assert output0 == output1


class TestSpanClassificationBehavior:
    """"""

    @staticmethod
    def predict_fn(list_text: List[str]):
        return [[Span(start=0, end=10, label=1), Span(start=20, end=30, label=1)], ] * len(list_text)

    def test_run(self, text_sample, random_span):
        n_samples = 5
        behavior = SpanClassificationBehavior(
            capability="Capability 1",
            name="Test span classification",
            test_type=BehaviorType.invariance,
            samples=[text_sample] * n_samples,
            labels=[[random_span, ] * 4] * n_samples,
            predict_fn=self.predict_fn
        )
        behavior.run()
        assert len(behavior.outputs) == n_samples
        assert all([b.y_pred is not None for b in behavior.outputs])

        with pytest.raises(ValueError):
            behavior.run()

    def test_save_and_load(self, text_sample, random_span):
        """"""
        n_samples = 5
        behavior = SpanClassificationBehavior(
            capability="Capability 2",
            name="Test span classification",
            test_type=BehaviorType.invariance,
            samples=[text_sample] * n_samples,
            labels=[[random_span, ] * 4] * n_samples,
            predict_fn=self.predict_fn
        )
        behavior.run()
        behavior.to_file("tmp_data/")
        output0 = behavior.outputs

        new_behavior = SpanClassificationBehavior.from_file(
            "tmp_data/Test_span_classification.pkl",
            self.predict_fn
        )
        new_behavior.run()
        output1 = behavior.outputs

        assert output0 == output1


class TestTokenClassificationBehavior:
    """"""

    @staticmethod
    def predict_fn(list_text: List[str]):
        preds = []
        for text in list_text:
            tokens = text.split()
            y = [Token(pos=i, label=i % 2) for i in range(len(tokens))]
            preds.append(y)
        return preds

    def test_run(self, text_sample):
        n_samples = 5
        tokens = text_sample.split()
        labels = [[Token(pos=i, label=i % 3) for i in range(len(tokens))], ] * n_samples

        behavior = TokenClassificationBehavior(
            capability="Capability 1",
            name="Test token classification",
            test_type=BehaviorType.invariance,
            samples=[text_sample] * n_samples,
            labels=labels,
            predict_fn=self.predict_fn
        )
        behavior.run()
        assert len(behavior.outputs) == n_samples
        assert all([b.y_pred is not None for b in behavior.outputs])

        with pytest.raises(ValueError):
            behavior.run()

    def test_save_and_load(self, text_sample):
        """"""
        n_samples = 5
        tokens = text_sample.split()
        labels = [[Token(pos=i, label=i % 3) for i in range(len(tokens))], ] * n_samples

        behavior = TokenClassificationBehavior(
            capability="Capability 2",
            name="Test token classification",
            test_type=BehaviorType.invariance,
            samples=[text_sample] * n_samples,
            labels=labels,
            predict_fn=self.predict_fn
        )
        behavior.run()
        behavior.to_file("tmp_data/")
        output0 = behavior.outputs

        new_behavior = TokenClassificationBehavior.from_file(
            "tmp_data/Test_token_classification.pkl",
            self.predict_fn
        )
        new_behavior.run()
        output1 = behavior.outputs

        assert output0 == output1
