from typing import Callable, Type

from alpaca_eval import annotators as ae_annotators
from alpaca_eval import utils as ae_utils


from . import completion_parsers


__all__ = ["BaseAnnotatorJSON", "SingleAnnotator"]



class BaseAnnotatorJSON(ae_annotators.BaseAnnotatorJSON):
    __doc__ = ae_annotators.BaseAnnotatorJSON.__doc__
    @property
    def SingleAnnotator(self) -> Type["ae_annotators.SingleAnnotator"]:
        return SingleAnnotator


class SingleAnnotator(ae_annotators.SingleAnnotator):
    __doc__ = ae_annotators.SingleAnnotator.__doc__

    def __init__(
        self,
        *args,
        is_store_raw_completions=True,  # let's save the raw completions. Useful when CoT or debugging
        **kwargs,
    ):
        super().__init__(*args, is_store_raw_completions=is_store_raw_completions, **kwargs)

    def _search_fn_completion_parser(self, name: str) -> Callable:
        try:
            return super()._search_fn_completion_parser(name)
        except AttributeError:
            # allows for new completion parsers
            return ae_utils.get_module_attribute(completion_parsers, name)