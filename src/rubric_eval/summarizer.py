"""Code for summarizing the rubric evaluation results."""
import abc
import copy
import json
import logging
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import pandas as pd
import yaml
from alpaca_eval import completion_parsers as ae_completion_parsers
from alpaca_eval import types as ae_types
from alpaca_eval import utils as ae_utils
from alpaca_eval.decoders import get_fn_completions

__all__ = ["summarize"]


def summarize(
    df: pd.DataFrame,
    summarizer_configs: str,
    base_dir: ae_types.AnyPath = Path(__file__).parent / "configs" / "summarizers_configs",
    **kwargs,
) -> dict[str, str]:
    """
    Summarize the rubric evaluation results.

    Args:
        df: DataFrame containing the rubric evaluation results.

        summarizer_configs : Path, optional
            Path to the directory relative to `base_dir` that contains all the summarizer configs. There should be a file
            named 'configs.yaml' in this directory that contains all the summarizer_kwargs needed to initialize the summarizer
            as well as a key "Summarizer" corresponding to the class name of the summarizer to use.

        base_dir : Path, optional
            Name of the base directory from which to search the summarizer_config directory.

        kwargs :
            Additional keyword arguments to pass to the summarizer.

    Returns:
        str: A summary of the rubric evaluation results.
    """
    logging.info(f"Creating the annotator from `{summarizer_configs}`.")
    configs_dir = Path(base_dir) / summarizer_configs

    path = configs_dir / "configs.yaml"
    with open(path, "r") as stream:
        try:
            configs = yaml.safe_load(stream)
        except yaml.YAMLError:
            logging.exception(f"Error when loading the configs file {path}:")

    Summarizer = globals()[configs["Summarizer"]]
    summarizer = Summarizer(configs_dir=configs_dir, base_dir=base_dir, **configs["summarizer_kwargs"], **kwargs)
    return summarizer(df)


class BaseSummarizer(abc.ABC):
    """Base class for summarizing the rubric grading.

    Args:
        configs_dir: Path to the directory containing the summarizer configs.
        summarize_by_cols: Columns to summarize by.
        approx_max_characters_to_summarize: Approximate maximum number of characters to summarize. The actual prompt
            size will be stochastic but shouldn't be more than 1.5 times this value (most likely less than 1.1 times).
        cache_path: Path to the cache file.
        fn_completion_parser: Function to use for parsing completions.
        completion_parser_kwargs: Keyword arguments for the completion parser.
        fn_completions (str): Function in `decoders.py` to use for decoding the output.
        completions_kwargs (dict, optional): kwargs for fn_completions. E.g. model_name, max_tokens, temperature, top_p, top_k, stop_seq.
    """

    def __init__(
        self,
        configs_dir: ae_types.AnyPath,
        summarize_by_cols: Sequence[str] = ("category",),
        approx_max_characters_to_summarize: Optional[int] = 70000,
        cache_path: Optional[str] = "auto",
        base_dir: ae_types.AnyPath = Path(__file__).parent / "configs" / "summarizers_configs",
        # completion arguments
        fn_completion_parser: Optional[Union[Callable, str]] = None,
        completion_parser_kwargs: Optional[dict[str, Any]] = None,
        fn_completions: Union[Callable, str] = "openai_completions",
        completions_kwargs: Optional[dict[str, Any]] = None,
        # prompting arguments
        add_columns_to_format: tuple[str] = ("instruction", "category"),
        completion_key: str = "summary",
    ):
        self.configs_dir = Path(configs_dir)
        self.summarize_by_cols = summarize_by_cols
        self.approx_max_characters_to_summarize = approx_max_characters_to_summarize
        self.summaries = {}
        self.cache_path = cache_path
        self.base_dir = base_dir
        self.completions_kwargs = completions_kwargs or {}

        self.add_columns_to_format = add_columns_to_format
        self.completion_key = completion_key

        # init completion function
        if self.cache_path == "auto":
            self.cache_path = self.configs_dir / "cache.json"
        if cache_path:
            cache_completions = get_fn_completions("cache_completions")
            self._fn_completions = partial(
                cache_completions,
                fn_completions=fn_completions,
                cache_path=self.cache_path,
                **self.completions_kwargs,
            )
        else:
            fn_completions = get_fn_completions(fn_completions)
            self._fn_completions = partial(fn_completions, **self.completions_kwargs)

        # init completion parser
        if fn_completion_parser is None:
            fn_completion_parser = lambda x: [x]
        elif isinstance(fn_completion_parser, str):
            fn_completion_parser = ae_utils.get_module_attribute(ae_completion_parsers, fn_completion_parser)
        completion_parser_kwargs = completion_parser_kwargs or {}
        self._fn_completion_parser = partial(fn_completion_parser, **completion_parser_kwargs)

    def __call__(self, df: pd.DataFrame, *args, **kwargs) -> dict[str, str]:
        summary_completions = self.summarize(df)
        summary_completions["summarizer"] = self.configs_dir.name
        return summary_completions

    @abc.abstractmethod
    def summarize(self, df: pd.DataFrame) -> dict[str, str]:
        """Summarize the rubric evaluation results."""
        ...

    def _generate_and_parse(self, prompt) -> dict[str, Any]:
        """Helper function to generate prompts, get completions and parse them."""
        completions = self._fn_completions(prompts=[prompt])
        assert len(completions) == 1
        completions = {
            k: v[0] if (isinstance(v, (list, tuple)) and len(v) == 1) else v for k, v in completions[0].items()
        }
        completions["date"] = datetime.now().isoformat()
        completions["version"] = ae_utils.get_multi_package_version(["rubric_eval", "alpaca_eval"])
        completions = {(f"summarizer_{k}" if k != "completions" else k): v for k, v in completions.items()}
        parsed_completions = self._fn_completion_parser(completions.pop("completions"))
        assert len(parsed_completions) == 1
        completions[self.completion_key] = parsed_completions[0]
        return completions


class UnstructuredSummarizer(BaseSummarizer):
    def __init__(
        self,
        *args,
        cols_to_split_by: Sequence[str] = ("category",),
        prompt_template: ae_types.AnyPath,
        subsequent_prompt_template: Optional[ae_types.AnyPath] = None,
        is_add_rubric_description: bool = True,
        example_delimiter: str = "-----------------\n\n",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cols_to_split_by = cols_to_split_by
        self.is_add_rubric_description = is_add_rubric_description
        self.initial_prompt_template = ae_utils.read_or_return(prompt_template, relative_to=self.base_dir)
        subsequent_prompt_template = subsequent_prompt_template or prompt_template
        self.subsequent_prompt_template = ae_utils.read_or_return(subsequent_prompt_template, relative_to=self.base_dir)
        self.example_delimiter = example_delimiter

    def summarize(self, df: pd.DataFrame, *args, **kwargs) -> dict[str, str]:
        """Summarize the rubric evaluation results in an unstructured format."""
        splits = self._split_dfs_by_cols(df)
        if len(splits) == 1:
            # Summarize df using potential hierarchy if doesn't fit in approx_max_characters_to_summarize
            summary_completions = self._get_summary(df)
        else:
            # 1. Summarize each split (e.g. category) separately using potential hierarchy
            all_summaries = dict()
            for val_splitted_by, split_df in splits:
                all_summaries[val_splitted_by] = self._get_summary(split_df)

            # 2. Summarize the splitted summaries
            summary_completions = self._get_subsequent_summary(list(all_summaries.values()), n_examples=len(df))
            summary_completions["summaries_splitted"] = all_summaries

        return summary_completions

    def _get_summary(self, df: pd.DataFrame) -> dict[str, str]:
        """Get the summary of the DataFrame."""

        # initialize summary
        summaries = []
        formatted_cols = list(self.add_columns_to_format) + ["evaluation"]
        for split_df in self._split_dfs_by_lengths(
            df, formatted_cols=formatted_cols, prompt_template=self.initial_prompt_template
        ):
            prompt = self._make_initial_prompt(split_df)
            summaries.append(self._generate_and_parse(prompt))

        return self._get_subsequent_summary(summaries, n_examples=len(df))

    def _get_subsequent_summary(self, summaries: list[dict[str, Any]], n_examples: int) -> dict[str, str]:
        """Summarize a dataframe of summaries, where the summary is potentially hierarchical in case the
        prompt is longer than `approx_max_characters_to_summarize`.
        """
        total_time = 0
        total_price = 0

        if len(summaries) < 1:
            raise ValueError("Can't summarize fewer than one summaries.")

        while len(summaries) > 1:
            df_summaries = pd.DataFrame.from_records(summaries)
            total_price += df_summaries["summarizer_price_per_example"].sum()
            total_time += df_summaries["summarizer_time_per_example"].sum()
            summaries = []
            for split_df in self._split_dfs_by_lengths(
                df_summaries, formatted_cols=[self.completion_key], prompt_template=self.subsequent_prompt_template
            ):
                prompt = self._make_prompt(
                    split_df, self.subsequent_prompt_template, add_columns_to_format=[self.completion_key]
                )
                summaries.append(self._generate_and_parse(prompt))

        out = summaries[0]
        out["summarizer_price_per_example"] = (out["summarizer_price_per_example"] + total_price) / n_examples
        out["summarizer_time_per_example"] = (total_time + out["summarizer_time_per_example"]) / n_examples
        return out

    def _split_dfs_by_cols(self, df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
        """Split the DataFrame based on the columns to split by, if it's in the dataframe.

        Args:
            df: DataFrame to split.

        Returns:
            list[pd.DataFrame]: List of splitted DataFrames.
        """
        cols_to_split_by = [col for col in self.cols_to_split_by if col in df.columns]
        if len(cols_to_split_by) == 0:
            splitted_dfs = [("", df)]
        else:
            splitted_dfs = [
                # return a tuple. THe first element is the joined string of the values of the columns to split by
                # The second element is the DataFrame
                (", ".join(str(v) for v in vals_splitted_by), group_df)
                for vals_splitted_by, group_df in df.groupby(cols_to_split_by)
            ]
            if len(splitted_dfs) == 1:
                splitted_dfs = [("", df)]
            else:
                logging.info(
                    f"Splitting the DataFrame into {len(splitted_dfs)} parts based on {self.cols_to_split_by} for hierarchical summarization."
                )
        return splitted_dfs

    def _split_dfs_by_lengths(
        self, df: pd.DataFrame, formatted_cols: Sequence[str], prompt_template: str
    ) -> list[pd.DataFrame]:
        """
        Split the dataframe in such a ways that the columns to format have approx less than
        `self.approx_max_characters_to_summarize` characters.
        """
        splitted_dfs = []

        # Calculate the string length of the formatted columns
        formatted_str = df[formatted_cols].to_string(index=False)

        max_char = int((self.approx_max_characters_to_summarize - len(prompt_template)) * 0.9)

        # If the formatted string length is within the limit, append the whole group
        if len(formatted_str) <= max_char:
            splitted_dfs.append(df)
        else:
            # Calculate the number of splits and rows per split
            n_splits = len(formatted_str) // max_char + 1
            n_rows_per_split = len(df) // n_splits + 1
            if n_rows_per_split == 1:
                logging.warning(
                    "The number of evaluations to summarize is 1. Summarization of a single row is not useful, so we use 2 intead."
                    "Please consider increasing the `approx_max_characters_to_summarize` or the number of rows per split."
                )
                n_rows_per_split = 2

            # Split the group DataFrame by rows
            splitted_dfs.extend(df.iloc[i : i + n_rows_per_split] for i in range(0, len(df), n_rows_per_split))

            logging.info(
                f"Splitting the DataFrame into {len(splitted_dfs)} parts of approx {n_rows_per_split} rows for hierarchical summarization."
            )

        return splitted_dfs

    def _make_prompt(
        self,
        df: pd.DataFrame,
        prompt_template: str,
        add_columns_to_format: Sequence[str] = tuple(),
        fn_additional_to_format: Optional[Callable] = None,
    ) -> str:
        df = copy.deepcopy(df.reset_index())
        prompt_template = copy.deepcopy(prompt_template)

        to_format = copy.deepcopy(self.example_delimiter)
        for i, row in df.iterrows():
            for col in add_columns_to_format:
                value = row[col]
                if not isinstance(value, str):
                    value = json.dumps(value, indent=2)
                to_format += f"## {col.capitalize()} {i} \n{value}\n\n"

            if fn_additional_to_format:
                to_format += fn_additional_to_format(i, row)

            to_format += self.example_delimiter

        prompt = prompt_template.format(to_format=to_format)

        if len(prompt) > 1.5 * self.approx_max_characters_to_summarize:
            logging.warning(
                f"The prompt is much longer than `approx_max_characters_to_summarize={self.approx_max_characters_to_summarize}`..."
            )

        return prompt

    def _make_initial_prompt(self, df: pd.DataFrame) -> str:
        def fn_additional_to_format(i, row):
            """Additional formatting for rubric."""
            evaluation_dict = {e["criterion"]: e for e in row["evaluation"]}
            for rubric_per_crit in row["rubric"]:  # list of dict
                curr_eval = evaluation_dict[rubric_per_crit["criterion"]]
                curr_eval["criteria_weight"] = rubric_per_crit["weight"]
                if self.is_add_rubric_description:
                    desc = rubric_per_crit["performance_to_description"][curr_eval["performance"]]
                    curr_eval["rubric_description"] = desc

            value = json.dumps(list(evaluation_dict.values()), indent=2)
            return f"## Rubric-based evaluation {i} \n{value}\n\n"

        return self._make_prompt(
            df,
            self.initial_prompt_template,
            add_columns_to_format=self.add_columns_to_format,
            fn_additional_to_format=fn_additional_to_format,
        )


class StructuredSummarizer(BaseSummarizer):
    def __init__(
        self,
        *args,
        apply_structured_summary_at_each_level: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.apply_structured_summay_at_each_level = apply_structured_summary_at_each_level

    def summarize(self, df: pd.DataFrame, *args, **kwargs) -> dict[str, str]:
        """
        Summarize the rubric evaluation results in a structured format.

        Args:
            df: DataFrame containing the rubric evaluation results.

        Returns:
            str: A structured summary of the rubric evaluation results.
        """
        ...
