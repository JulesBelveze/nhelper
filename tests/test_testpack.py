import pytest

from nlptest.behavior import SequenceClassificationBehavior, DuplicateBehaviorError
from nlptest.performers import Performer
from nlptest.testpack import TestPack, PyTorchTestPack
from nlptest.types import BehaviorType


@pytest.fixture
def seq_classification_behavior():
    return SequenceClassificationBehavior(
        capability="Capability 1",
        name="Test sequence classification",
        test_type=BehaviorType.invariance,
        samples=["TEST"],
        labels=[1],
        predict_fn=lambda x: [1, ] * len(x)
    )


@pytest.fixture
def seq_classification_behavior2():
    return SequenceClassificationBehavior(
        capability="Capability 2",
        name="Test sequence classification 2",
        test_type=BehaviorType.invariance,
        samples=["TEST"],
        labels=[2],
        predict_fn=lambda x: [1, ] * len(x)
    )


@pytest.fixture
def performer():
    return Performer()


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

    def test_run(self, seq_classification_behavior, seq_classification_behavior2, performer):
        """"""
        testpack = TestPack(performer=performer)
        testpack.add([seq_classification_behavior, seq_classification_behavior2])

        testpack.run()
        with pytest.raises(ValueError):
            testpack.run()

    def test_save_and_load(self, seq_classification_behavior, seq_classification_behavior2, performer):
        """"""
        testpack = TestPack(performer=performer)
        testpack.add([seq_classification_behavior, seq_classification_behavior2])
        testpack.run()
        outputs1 = testpack.outputs
        testpack.to_file("/tmp/test")

        testpack2 = TestPack.from_file(
            "/tmp/test",
            lambda x: [1, ] * len(x),
            Performer()
        )
        testpack2.run()
        outputs2 = testpack2.outputs

        assert outputs1 == outputs2


class TestPyTorchTestPack:
    """"""

    @staticmethod
    def identity(**kwargs):
        return kwargs

    def test_instantiation(self, seq_classification_behavior, seq_classification_behavior2):
        """"""
        testpack = PyTorchTestPack(
            capabilities=[seq_classification_behavior.capability, seq_classification_behavior2.capability],
            names=[seq_classification_behavior.name, seq_classification_behavior2.name],
            test_types=[seq_classification_behavior.test_type, seq_classification_behavior2.test_type],
            texts=seq_classification_behavior.samples + seq_classification_behavior2.samples,
            labels=seq_classification_behavior.labels + seq_classification_behavior2.labels,
            processor=self.identity
        )

        assert list(testpack[0].keys()) == ["capability", "name", "test_type", "text", "labels"]

    def test_from_testpack(self, seq_classification_behavior):
        """"""
        pt_testpack = PyTorchTestPack(
            capabilities=[seq_classification_behavior.capability],
            names=[seq_classification_behavior.name],
            test_types=[seq_classification_behavior.test_type],
            texts=seq_classification_behavior.samples,
            labels=seq_classification_behavior.labels,
            processor=self.identity
        )

        testpack = TestPack()
        testpack.add(seq_classification_behavior)

        pt_testpack_from_testpack = PyTorchTestPack.from_testpack(testpack, self.identity)
        assert sorted([elt for elt in pt_testpack_from_testpack], key=lambda d: d["name"]) == \
               sorted([elt for elt in pt_testpack], key=lambda d: d["name"])

    def test_from_behaviors(self, seq_classification_behavior, seq_classification_behavior2):
        """"""
        testpack = TestPack()
        testpack.add([seq_classification_behavior, seq_classification_behavior2])
        testpack.to_file("/tmp/test")

        pt_testpack = PyTorchTestPack.from_saved_behaviors("/tmp/test", processor=self.identity)

        pt_testpack2 = PyTorchTestPack(
            capabilities=[seq_classification_behavior.capability, seq_classification_behavior2.capability],
            names=[seq_classification_behavior.name, seq_classification_behavior2.name],
            test_types=[seq_classification_behavior.test_type, seq_classification_behavior2.test_type],
            texts=seq_classification_behavior.samples + seq_classification_behavior2.samples,
            labels=seq_classification_behavior.labels + seq_classification_behavior2.labels,
            processor=self.identity
        )

        assert sorted([elt for elt in pt_testpack2], key=lambda d: d["name"]) == \
               sorted([elt for elt in pt_testpack], key=lambda d: d["name"])
