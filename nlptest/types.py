from enum import Enum
from typing import Union, List

from pydantic import BaseModel


class BehaviorType(str, Enum):
    """"""
    invariance = "invariance"
    directional = "directional"
    minimum_functionality = "minimum functionality"


class TaskType(str, Enum):
    """"""
    sequence_classification = "sequence_classification"
    target_sequence_classification = "targeted_sequence_classification"
    span_classification = "span_classification"
    token_classification = "token_classification"


class Span(BaseModel):
    """Representation of a 'Span' object for span classification tasks"""
    start: int
    end: int
    label: Union[str, int]
    text: str = None
    prob: float = None

    def __str__(self):
        return str(self.start) + "_" + str(self.end) + "_" + str(self.label)

    def __hash__(self):
        return hash(str(self))

    def __lt__(self, other):
        return str(self) < str(other)

    def __eq__(self, other):
        if other.__class__ is self.__class__:
            return str(self.start) + "_" + str(self.end) + "_" + str(self.label) == \
                   str(other.start) + "_" + str(other.end) + "_" + str(other.label)


class SpanClassificationOutput(BaseModel):
    """Output of a span classification model"""
    text: str
    y_pred: List[Span]
    y: List[Span]

    @property
    def success(self):
        if len(self.y) != len(self.y_pred):
            return False

        return sorted(self.y) == sorted(self.y_pred)


class SequenceClassificationOutput(BaseModel):
    """Output of a sequence classification model"""
    text: str
    y_pred: Union[str, int]
    y_pred_prob: float = None
    y: Union[str, int]

    @property
    def success(self):
        return self.y == self.y_pred


class TargetedSequenceClassificationOutput(SequenceClassificationOutput):
    """Output of a targeted sequence classification model"""
    target: str


class MultiLabelSequenceClassificationOutput(BaseModel):
    """"""
    text: str
    y_pred: List[int]
    y_pred_prob: List[float] = None
    y: List[int]

    @property
    def success(self):
        return self.y == self.y_pred


class Token(BaseModel):
    """"""
    pos: int
    prob: float = None
    label: int


class TokenClassificationOutput(BaseModel):
    """"""
    text: str
    y_pred: Union[List[Token], List[int]]
    y_pred_prob: List[float] = None
    y: Union[List[Token], List[int]]

    @property
    def success(self):
        return self.y == self.y_pred


BehaviorOutput = Union[
    SpanClassificationOutput,
    SequenceClassificationOutput,
    TargetedSequenceClassificationOutput,
    MultiLabelSequenceClassificationOutput,
    TokenClassificationOutput
]
