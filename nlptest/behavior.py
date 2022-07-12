import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import List, Union, Callable, Optional, Any

from overrides import overrides

from .types import BehaviorType, TaskType, SequenceClassificationOutput, Span, SpanClassificationOutput, \
    MultiLabelSequenceClassificationOutput, Token, TokenClassificationOutput


class Behavior(object):
    """Model's Behavior to be tested"""

    def __init__(self, capability: str, name: str, test_type: BehaviorType, task_type: TaskType, samples: List[str],
                 labels: Any, predict_fn: Callable = None, description: str = None):
        """
        :param capability: capability to test
        :param name: behavior name (used for identification)
        :param test_type: type of test
        :param task_type: type of task
        :param samples: list of samples to test
        :param predict_fn: function used for prediction
        :param labels: set of labels
        :param description: behavior's description
        """
        if isinstance(labels, list):
            assert len(labels) == len(samples), \
                "Provide either a single label or one label per sample"
        self.capability = capability
        self.name = name
        self.test_type = test_type
        self.task_type = task_type
        self.description = description

        self._predict_fn = predict_fn
        self.samples = samples
        self.labels = labels

        self._is_ran = False
        self.outputs = []

    @property
    def predict_fn(self):
        return self._predict_fn

    @predict_fn.setter
    def predict_fn(self, value):
        self._predict_fn = value

    def run(self) -> None:
        """"""
        raise NotImplementedError()

    def reset(self) -> None:
        """"""
        self.outputs = []
        self._is_ran = False

    def to_file(self, path_folder: str) -> None:
        """Save the Behavior as a pickle object"""
        Path(path_folder).mkdir(parents=True, exist_ok=True)

        file_name = "_".join(self.name.split())
        path = Path(os.path.join(path_folder, f"{file_name}.pkl"))

        self_copy = deepcopy(self)
        self_copy.reset()
        self_copy.predict_fn = None

        with open(path, "wb") as writer:
            pickle.dump(self_copy, writer)

    @classmethod
    def from_file(cls, path_to_file: str, predict_fn: Callable = None):
        """Loads a Behavior from a pickle file"""
        with open(path_to_file, "rb") as reader:
            behavior = pickle.load(reader)

        behavior.predict_fn = predict_fn
        return behavior


class SequenceClassificationBehavior(Behavior):
    """"""

    def __init__(self, capability: str, name: str, test_type: BehaviorType, samples: List[str],
                 labels: Union[Union[str, int], List[Union[str, float]]], predict_fn: Callable = None,
                 description: str = None):
        """
        :param capability:
        :param name:
        :param test_type:
        :param samples:
        :param predict_fn:
        :param labels:
        :param description:
        """
        super().__init__(capability, name, test_type, TaskType.sequence_classification, samples, labels, predict_fn,
                         description)

    @overrides
    def run(self) -> None:
        """"""
        if self._is_ran:
            raise ValueError(f"This 'Behavior' has already been ran.")
        predictions = self.predict_fn(self.samples)

        for prediction, truth, text in zip(predictions, self.labels, self.samples):
            if isinstance(prediction, tuple) or isinstance(prediction, list):
                y_pred, prob = prediction
            else:
                y_pred = prediction
                prob = None
            self.outputs.append(
                SequenceClassificationOutput(
                    text=text,
                    y_pred=y_pred,
                    y_pred_prob=prob,
                    y=truth
                )
            )
        self._is_ran = True

    def __str__(self):
        return f"<SequenceClassificationBehavior: name='{self.name}'>"


class MultiLabelSequenceClassificationBehavior(Behavior):
    """"""

    def __init__(self, capability: str, name: str, test_type: BehaviorType, samples: List[str],
                 labels: Union[List[int], List[List[int]]], predict_fn: Callable = None, description: str = None):
        """
        :param capability
        :param name:
        :param test_type:
        :param samples:
        :param predict_fn:
        :param labels:
        :param description:
        """
        super().__init__(capability, name, test_type, TaskType.sequence_classification, samples, labels, predict_fn,
                         description)

    @overrides
    def run(self) -> None:
        """"""
        if self._is_ran:
            raise ValueError(f"This 'Behavior' has already been ran.")
        predictions = self.predict_fn(self.samples)

        for prediction, truth, text in zip(predictions, self.labels, self.samples):
            if isinstance(prediction, tuple):
                y_pred, prob = prediction
            else:
                y_pred = prediction
                prob = None
            self.outputs.append(
                MultiLabelSequenceClassificationOutput(
                    text=text,
                    y_pred=y_pred,
                    y_pred_prob=prob,
                    y=truth
                )
            )
        self._is_ran = True

    def __str__(self):
        return f"<MultiLabelSequenceClassificationBehavior: name='{self.name}'>"


class SpanClassificationBehavior(Behavior):
    """"""

    def __init__(self, capability: str, name: str, test_type: BehaviorType, samples: List[str],
                 labels: List[List[Optional[Span]]], predict_fn: Callable = None, description: str = None):
        """
        :param capability:
        :param name:
        :param test_type:
        :param samples:
        :param predict_fn:
        :param labels:
        :param description:
        """
        super().__init__(capability, name, test_type, TaskType.span_classification, samples, labels, predict_fn,
                         description)

    @overrides
    def run(self) -> None:
        """"""
        if self._is_ran:
            raise ValueError(f"This 'Behavior' has already been ran.")

        predictions = self.predict_fn(self.samples)

        for predicted_spans, true_spans, text in zip(predictions, self.labels, self.samples):
            sample_spans = []
            if len(predicted_spans) > 0 and isinstance(predicted_spans[0], tuple):
                for span in predicted_spans:
                    if len(span) >= 3:
                        span = Span(**{key: span[i] for i, key in enumerate(Span.__fields__.keys())})
                    else:
                        raise ValueError(
                            f"Output of type 'Span' requires at least 3 elements, got {len(span)} instead.")
                    sample_spans.append(span)
            elif len(predicted_spans) > 0 and isinstance(predicted_spans[0], Span):
                sample_spans = predicted_spans
            elif len(predicted_spans) > 0:
                raise ValueError(
                    f"Expected span prediction to be of type 'tuple' or 'Span' got '{type(predicted_spans[0])}'")
            self.outputs.append(
                SpanClassificationOutput(
                    text=text,
                    y_pred=sample_spans,
                    y=true_spans
                )
            )

        self._is_ran = True

    def __str__(self):
        return f"<SpanClassificationBehavior: name='{self.name}'>"


class TokenClassificationBehavior(Behavior):
    """"""

    def __init__(self, capability: str, name: str, test_type: BehaviorType, samples: List[str],
                 labels: List[Union[List[Token], List[int]]], predict_fn: Callable = None, description: str = None):
        """
        :param capability:
        :param name:
        :param test_type:
        :param samples:
        :param predict_fn:
        :param labels:
        :param description:
        """
        super().__init__(capability, name, test_type, TaskType.token_classification, samples, labels, predict_fn,
                         description)

    @overrides
    def run(self) -> None:
        """"""
        if self._is_ran:
            raise ValueError(f"This 'Behavior' has already been ran.")

        predictions = self.predict_fn(self.samples)

        for predicted_tokens, true_tokens, text in zip(predictions, self.labels, self.samples):
            if isinstance(predicted_tokens[0], Token):
                sample_tokens_pred = predicted_tokens
            elif isinstance(predicted_tokens[0], int):
                sample_tokens_pred = [Token(pos=i, label=l) for i, l in enumerate(predicted_tokens)]
            else:
                raise ValueError(
                    f"Expected token prediction to be of type 'int' or 'Token' got '{type(predicted_tokens[0])}'")

            if isinstance(true_tokens[0], Token):
                sample_tokens = true_tokens
            elif isinstance(true_tokens[0], int):
                sample_tokens = [Token(pos=i, label=l) for i, l in enumerate(true_tokens)]
            else:
                raise ValueError(
                    f"Expected token prediction to be of type 'int' or 'Token' got '{type(true_tokens[0])}'")

            self.outputs.append(
                TokenClassificationOutput(
                    text=text,
                    y_pred=sample_tokens_pred,
                    y=sample_tokens
                )
            )

        self._is_ran = True

    def __str__(self):
        return f"<TokenClassificationBehavior: name='{self.name}'>"


class DuplicateBehaviorError(Exception):
    pass


class BehaviorSet(set):
    """"""

    def add(self, value: Behavior):
        """"""
        if value in self:
            raise DuplicateBehaviorError(f"Behavior '{value}' already present in set.")
        super().add(value)

    def update(self, values: List[Behavior]):
        """"""
        error_values = []
        for value in values:
            if value in self:
                error_values.append(value)
        if error_values:
            raise DuplicateBehaviorError(f"Behavior(s) '{error_values}' already present in set.")
        super().update(values)
