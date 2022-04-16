from typing import List

from transformers import pipeline


class Generator(object):
    def __init__(self, fill_mask_model_name: str = None, translator_model_name: str = None):
        self.fill_mask_model_name = fill_mask_model_name
        self.translator_model_name = translator_model_name

        self.fill_pipeline = pipeline("fill-mask", model=fill_mask_model_name) if fill_mask_model_name else None
        self.translator_pipeline = pipeline("text2text-generation",
                                            model=translator_model_name) if translator_model_name else None

    def fill_mask(self, template: str, top_k: int):
        """"""
        if self.fill_mask_model_name is None:
            raise ValueError("The Generator has not been instantiated with a 'fill_mask_model_name'.")

        if "roberta" in self.fill_mask_model_name:
            template = template.replace("[MASK]", "<mask>")
        unmasked = self.fill_pipeline(template, top_k=top_k)
        return [pred["sequence"] for pred in unmasked]

    def translate(self, template: str):
        """"""
        if self.translator_model_name is None:
            raise ValueError("The Generator has not been instantiated with a 'translator_model_name'.")

        translation = self.translator_pipeline(template)
        return translation[0]["generated_text"]

    @staticmethod
    def generate(template: str, **kwargs) -> List[str]:
        """"""
        assert max(map(len, kwargs.values())) == min(map(len, kwargs.values())), \
            "Please provide the same number number of alternatives for all keywords."

        flatten_kwargs = [dict(zip(kwargs, t)) for t in zip(*kwargs.values())]

        generations = []
        for combination in flatten_kwargs:
            generations.append(template.format(**combination))

        return generations
