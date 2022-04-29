from typing import List, Union, Callable, Optional, Any

from overrides import overrides

from .types import BehaviorType, TaskType, SequenceClassificationOutput, Span, SpanClassificationOutput


class Behavior(object):
    """"""

    def __init__(self, test_type: BehaviorType, task_type: TaskType, samples: List[str], predict_fn: Callable,
                 labels: Any, description: str = None):
        if isinstance(labels, list):
            assert len(labels) == len(samples), \
                "Provide either a single label or one label per sample"
        self.test_type = test_type
        self.task_type = task_type
        self.description = description

        self.predict_fn = predict_fn
        self.samples = samples
        self.labels = labels

    def run(self):
        """"""
        raise NotImplementedError()


class SequenceClassificationBehavior(Behavior):
    """"""

    def __init__(self, test_type: BehaviorType, task_type: TaskType, samples: List[str], predict_fn: Callable,
                 labels: Union[Union[str, int], List[Union[str, float]]], description: str = None):
        super().__init__(test_type, task_type, samples, predict_fn, labels, description)

    @overrides
    def run(self):
        """"""
        predictions = self.predict_fn(self.samples)
        outputs = []
        for prediction, truth, text in zip(predictions, self.labels, self.samples):
            if isinstance(prediction, tuple) or isinstance(prediction, list):
                y_pred, prob = prediction
            else:
                y_pred = prediction
                prob = None
            outputs.append(
                SequenceClassificationOutput(
                    text=text,
                    y_pred=y_pred,
                    y_pred_prob=prob,
                    y=truth
                )
            )
        return outputs


class SpanClassificationBehavior(Behavior):
    """"""

    def __init__(self, test_type: BehaviorType, task_type: TaskType, samples: List[str], predict_fn: Callable,
                 labels: List[List[Optional[Span]]], description: str = None):
        super().__init__(test_type, task_type, samples, predict_fn, labels, description)

    @overrides
    def run(self):
        predictions = self.predict_fn(self.samples)
        outputs = []
        print(predictions)
        for predicted_spans, true_spans, text in zip(predictions, self.labels, self.samples):
            sample_spans = []
            if isinstance(predicted_spans[0], tuple):
                for span in predicted_spans:
                    if len(span) >= 3:
                        span = Span(**{key: span[i] for i, key in enumerate(Span.__fields__.keys())})
                    else:
                        raise ValueError(
                            f"Output of type 'Span' requires at least 3 elements, got {len(span)} instead.")
                    sample_spans.append(span)
            elif isinstance(predicted_spans[0], Span):
                sample_spans = predicted_spans
            else:
                raise ValueError(
                    f"Expected span prediction to be of type 'tuple' or 'Span' got '{type(predicted_spans[0])}'")
            outputs.append(
                SpanClassificationOutput(
                    text=text,
                    y_pred=sample_spans,
                    y=true_spans
                )
            )
        return outputs


class DuplicateBehaviorError(Exception):
    pass


class BehaviorSet(set):
    def add(self, value: Behavior):
        if value in self:
            raise DuplicateBehaviorError(f"Behavior '{value}' already present in set.")
        super().add(value)

    def update(self, values: List[Behavior]):
        error_values = []
        for value in values:
            if value in self:
                error_values.append(value)
        if error_values:
            raise DuplicateBehaviorError(f"Behavior(s) '{error_values}' already present in set.")
        super().update(values)
