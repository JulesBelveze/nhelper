from nhelper.behavior import SequenceClassificationBehavior, SpanClassificationBehavior, TokenClassificationBehavior
from nhelper.performers import Performer
from nhelper.types import BehaviorType, Span, Token


class TestPerformer:
    """"""

    def test_metrics_seq_classification(self):
        """"""
        seq_classification_behavior = SequenceClassificationBehavior(
            capability="Capability 1",
            name="Test sequence classification",
            test_type=BehaviorType.invariance,
            samples=["This is a test"],
            labels=[1],
            predict_fn=lambda x: [1] * len(x)
        )
        seq_classification_behavior2 = SequenceClassificationBehavior(
            capability="Capability 2",
            name="Test sequence classification 2",
            test_type=BehaviorType.directional,
            samples=["This is a test", "This is a 2nd test"],
            labels=[2, 1],
            predict_fn=lambda x: [1] * len(x)
        )
        performer = Performer(
            metric_type="weighted"
        )
        performer.fit([seq_classification_behavior, seq_classification_behavior2])
        print(performer.result)
        assert performer.result[f"Behavior type - {BehaviorType.directional.value}"] == \
               performer.result[f"Name - {seq_classification_behavior2.name}"] == [0.5, '1/2']
        assert performer.result[f"Behavior type - {BehaviorType.invariance.value}"] == \
               performer.result[f"Name - {seq_classification_behavior.name}"] == [1.0, '1/1']
        assert performer.result["Total"] == [2 / 3, '2/3']

    def test_metrics_span_classification(self):
        """"""
        span_classification_behavior = SpanClassificationBehavior(
            capability="Capability 1",
            name="Test span classification",
            test_type=BehaviorType.invariance,
            samples=["This is a test"],
            labels=[[Span(start=0, end=10, label=1)]],
            predict_fn=lambda x: [[Span(start=0, end=10, label=1)], ] * len(x)
        )
        span_classification_behavior2 = SpanClassificationBehavior(
            capability="Capability 2",
            name="Test span classification 2",
            test_type=BehaviorType.directional,
            samples=["This is a test", "This is a 2nd test"],
            labels=[
                [Span(start=0, end=8, label=0)],
                [Span(start=2, end=5, label=1), Span(start=1, end=8, label=0)]
            ],
            predict_fn=lambda x: [
                [Span(start=0, end=4, label=1)],
                [Span(start=1, end=2, label=0), Span(start=2, end=5, label=1), Span(start=4, end=6, label=1)]
            ]
        )
        performer = Performer(
            metric_type="weighted"
        )
        performer.fit([span_classification_behavior, span_classification_behavior2])

        assert performer.result[f"Behavior type - {BehaviorType.directional.value}"] == \
               performer.result[f"Name - {span_classification_behavior2.name}"] == [0.0, '0/2']
        assert performer.result[f"Behavior type - {BehaviorType.invariance.value}"] == \
               performer.result[f"Name - {span_classification_behavior.name}"] == [1.0, '1/1']
        assert performer.result["Total"] == [1 / 3, '1/3']

    def test_metrics_span_classification_binarized(self):
        """"""
        span_classification_behavior = SpanClassificationBehavior(
            capability="Capability 1",
            name="Test span classification",
            test_type=BehaviorType.invariance,
            samples=["This is a test"],
            labels=[[Span(start=0, end=10, label=4)]],
            predict_fn=lambda x: [[Span(start=0, end=10, label=5)], ] * len(x)
        )
        span_classification_behavior2 = SpanClassificationBehavior(
            capability="Capability 2",
            name="Test span classification 2",
            test_type=BehaviorType.directional,
            samples=["This is a test", "This is a 2nd test"],
            labels=[
                [Span(start=2, end=5, label=10), Span(start=1, end=8, label=10)],
                [Span(start=2, end=5, label=1), Span(start=1, end=8, label=0)]
            ],
            predict_fn=lambda x: [
                [Span(start=2, end=5, label=6), Span(start=1, end=8, label=10)],
                [Span(start=2, end=2, label=0), Span(start=1, end=8, label=0), Span(start=4, end=6, label=1)]
            ]
        )
        performer = Performer(
            metric_type="weighted",
            binarize=True
        )
        performer.fit([span_classification_behavior, span_classification_behavior2])

        assert performer.result[f"Behavior type - {BehaviorType.directional.value}"] == \
               performer.result[f"Name - {span_classification_behavior2.name}"] == [0.5, '1/2']
        assert performer.result[f"Behavior type - {BehaviorType.invariance.value}"] == \
               performer.result[f"Name - {span_classification_behavior.name}"] == [1.0, '1/1']
        assert performer.result["Total"] == [2 / 3, '2/3']

    def test_metrics_token_classification(self):
        """"""
        token_classification_behavior = TokenClassificationBehavior(
            capability="Capability 1",
            name="Test token classification",
            test_type=BehaviorType.invariance,
            samples=["This is a test"],
            labels=[[Token(pos=0, label=0), Token(pos=1, label=0), Token(pos=2, label=0), Token(pos=3, label=1)]],
            predict_fn=lambda x: [[Token(pos=0, label=1), Token(pos=1, label=0), Token(pos=2, label=0),
                                   Token(pos=3, label=0)], ]
        )
        token_classification_behavior2 = TokenClassificationBehavior(
            capability="Capability 2",
            name="Test token classification 2",
            test_type=BehaviorType.directional,
            samples=["This is a test", "This is a 2nd test !"],
            labels=[
                [Token(pos=0, label=0), Token(pos=1, label=0), Token(pos=2, label=0), Token(pos=3, label=1)],
                [Token(pos=0, label=0), Token(pos=1, label=1), Token(pos=2, label=1), Token(pos=3, label=1),
                 Token(pos=4, label=1)]
            ],
            predict_fn=lambda x: [
                [Token(pos=0, label=1), Token(pos=1, label=0), Token(pos=2, label=0), Token(pos=3, label=1)],
                [Token(pos=0, label=0), Token(pos=1, label=1), Token(pos=2, label=1), Token(pos=3, label=1),
                 Token(pos=4, label=1)]
            ]
        )
        performer = Performer(
            metric_type="weighted"
        )
        performer.fit([token_classification_behavior, token_classification_behavior2])

        assert performer.result[f"Behavior type - {BehaviorType.directional.value}"] == \
               performer.result[f"Name - {token_classification_behavior2.name}"] == [0.5, '1/2']
        assert performer.result[f"Behavior type - {BehaviorType.invariance.value}"] == \
               performer.result[f"Name - {token_classification_behavior.name}"] == [0, '0/1']
        assert performer.result["Total"] == [1 / 3, '1/3']
