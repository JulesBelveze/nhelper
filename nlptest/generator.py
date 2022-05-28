from itertools import product
from typing import List, Union, Tuple

from transformers import pipeline


class Generator(object):
    """Helper object to create syntactical samples"""

    def __init__(self, fill_mask_model_name: str = None, translator_model_name: str = None):
        """

        :param fill_mask_model_name:
        :param translator_model_name:
        """
        self.fill_mask_model_name = fill_mask_model_name
        self.translator_model_name = translator_model_name

        self.fill_pipeline = pipeline("fill-mask", model=fill_mask_model_name) if fill_mask_model_name else None
        self.translator_pipeline = pipeline("text2text-generation",
                                            model=translator_model_name) if translator_model_name else None

    def fill_mask(self, template: Union[str, List[str]], top_k: int) -> List[str]:
        """
        Creates syntactical data by masking some words in a sentence and predicting
        which words should replace those masks.

        :param template: masked text(s) that will be used for prediction
        :param top_k: amount of syntactical texts to generate
        :return:
        """
        if self.fill_mask_model_name is None:
            raise ValueError("The Generator has not been instantiated with a 'fill_mask_model_name'.")

        if "roberta" in self.fill_mask_model_name:
            template = template.replace("[MASK]", "<mask>")
        unmasked = self.fill_pipeline(template, top_k=top_k)
        return [pred["sequence"] for pred in unmasked]

    def translate(self, template: Union[str, List[str]]) -> List[str]:
        """

        :param template:
        :return:
        """
        if self.translator_model_name is None:
            raise ValueError("The Generator has not been instantiated with a 'translator_model_name'.")

        translation = self.translator_pipeline(template)
        return [elt["generated_text"] for elt in translation]

    @staticmethod
    def generate(templates: Union[str, List[str]], generate_all: bool = False, return_pos: bool = False, **kwargs) -> \
            Union[List[str], Tuple[List[str], List[List[Tuple]]]]:
        """"""
        if isinstance(templates, str):
            templates = [templates]

        assert max(map(len, kwargs.values())) == min(map(len, kwargs.values())) or generate_all, \
            "Please provide the same number number of alternatives for all keywords or set 'generate_all' to True."

        if generate_all:
            combined_kwargs = [dict(zip(kwargs, t)) for t in product(*kwargs.values())]
        else:
            combined_kwargs = [dict(zip(kwargs, t)) for t in zip(*kwargs.values())]

        generations, all_positions = [], []
        for combination in combined_kwargs:
            for template in templates:
                text = template.format(**combination)
                generations.append(text)

                if return_pos:
                    positions = []
                    for label, word in combination.items():
                        start = text.find(word)
                        positions.append((start, start + len(word), label, word))
                    all_positions.append(positions)

        if return_pos:
            return generations, all_positions

        return generations
