import pytest
from nlptest.generator import Generator


@pytest.fixture()
def fill_mask_model_name():
    return "bert-base-uncased"


@pytest.fixture()
def fill_mask_model_name_roberta():
    return "distilroberta-base"


@pytest.fixture()
def translator_model_name():
    return "Helsinki-NLP/opus-mt-en-fr"


class TestGenerator:
    def test_mask_fill(self, fill_mask_model_name):
        """"""
        generator_fill = Generator(fill_mask_model_name=fill_mask_model_name)
        output = generator_fill.fill_mask("Hello I'm a [MASK] model.", top_k=1)
        assert len(output) == 1

        output = generator_fill.fill_mask("Hello I'm a [MASK] model.", top_k=5)
        assert len(output) == 5
        assert not any(["[MASK]" in pred for pred in output])

    def test_mask_fill_roberta(self, fill_mask_model_name_roberta):
        """"""
        generator_fill = Generator(fill_mask_model_name=fill_mask_model_name_roberta)
        output1 = generator_fill.fill_mask("Hello I'm a [MASK] model.", top_k=3)
        assert len(output1) == 3

        output2 = generator_fill.fill_mask("Hello I'm a <mask> model.", top_k=3)
        assert output1 == output2

    def test_translate(self, translator_model_name):
        """"""
        generator_translate = Generator(translator_model_name=translator_model_name)
        translation = generator_translate.translate("My name is Wolfgang and I live in Berlin")
        assert translation == "Je m'appelle Wolfgang et je vis à Berlin."

    def test_generate(self):
        """"""
        generator = Generator()
        generations = generator.generate(
            template="Hey my name is {name} {family_name}",
            name=["jules", "james", "john"],
            family_name=["a", "b", "c"]
        )
        assert generations == ["Hey my name is jules a", "Hey my name is james b", "Hey my name is john c"]

    def test_generate_bad_inputs(self):
        """"""
        generator = Generator()

        with pytest.raises(Exception) as e:
            generator.generate(
                template="Hey my name is {name} {family_name}",
                name=["jules"],
                family_name=["a", "b", "c"]
            )
