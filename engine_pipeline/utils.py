import os
import re
import sys
import copy
import math
import openai
import logging
import dataclasses
from openai import openai_object
from typing import Optional, Sequence, Union
import time

StrOrOpenAIObject = Union[str, openai_object.OpenAIObject]
openai.api_key = "your openai key"

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")

@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # suffix: Optional[str] = None
    # logprobs: Optional[int] = None
    # echo: bool = False

def openai_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: OpenAIDecodingArguments,
    model_name="gpt-4",
    sleep_time=2,
    batch_size=1,
    use_chat=False,
    max_instances=sys.maxsize,
    max_batches=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
) -> Union[
    Union[StrOrOpenAIObject],
    Sequence[StrOrOpenAIObject],
    Sequence[Sequence[StrOrOpenAIObject]],
]:
    """Decode with OpenAI API.

    Args:
        use_chat: weather use chat completion
        prompts: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        batch_size: Number of prompts to send in a single request. Only for non chat model.
        max_instances: Maximum number of prompts to decode.
        max_batches: Maximum number of batches to decode. This argument will be deprecated in the future.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """
    completions = []
    for prompt in prompts:
        batch_decoding_args = copy.deepcopy(decoding_args)
        shared_kwargs = dict(
                        model=model_name,
                        **batch_decoding_args.__dict__,
                        **decoding_kwargs,
                    )
        while True:
            try:
                messages = [{"role": "user", "content": prompt}]
                completion_batch = openai.ChatCompletion.create(
                        messages=messages, **shared_kwargs)
                choices = completion_batch.choices

                for choice in choices:
                    choice["total_tokens"] = completion_batch.usage.total_tokens
                completions.extend(choices)
                break
            except openai.error.OpenAIError as e:
                logging.warning(f"OpenAIError: {e}.")
                if "Please reduce your prompt" in str(e):
                    batch_decoding_args.max_tokens = int(
                        batch_decoding_args.max_tokens * 0.8
                    )
                    logging.warning(
                        f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying..."
                    )
                else:
                    logging.warning("Hit request rate limit; retrying...")
                    time.sleep(sleep_time)
    return completions