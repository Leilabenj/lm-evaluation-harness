import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.sglang_causallms import SGLangLM
from lm_eval.models.utils import (
    Collator,
    handle_stop_sequences,
    postprocess_generated_text,
)
from lm_eval.utils import simple_parse_args_string

eval_logger = logging.getLogger(__name__)

try:
    from pydantic import BaseModel, ValidationError
except ImportError:  # pragma: no cover - optional dependency
    BaseModel = None
    ValidationError = None
    eval_logger.warning(
        "Pydantic not installed. Schema validation will be skipped. "
        "Install with: pip install pydantic"
    )


@register_model("sglang-schema")
class SGLangSchemaLM(SGLangLM):
    """
    Thin schema-aware wrapper around `SGLangLM`.

    Instead of mirroring the full harness `generate_until()` logic we lean on the
    native structured output API described in https://docs.sglang.io/advanced_features/structured_outputs.html
    (see `sgl.json(...)`). SGLang enforces the JSON schema during sampling, so this
    class only needs to:
      * forward the schema through sampling params (`json_schema`)
      * optionally run a light Pydantic validation pass
    """
    
    def __init__(
        self,
        pretrained: str,
        schema_model: Optional[type] = None,
        response_schema: Optional[Union[Dict, str]] = None,
        schema_file: Optional[str] = None,
        validate_with_pydantic: bool = True,
        strict_validation: bool = False,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        self.schema_model = schema_model
        self.strict_validation = strict_validation
        self.validate_with_pydantic = bool(
            validate_with_pydantic and schema_model and BaseModel is not None
        )
        self.base_url = base_url or kwargs.pop("base_url", None)
        self.use_remote_api = self.base_url is not None

        if self.schema_model and BaseModel is None:
            raise ValueError(
                "schema_model requires Pydantic. Install it or set validate_with_pydantic=False."
            )

        self.response_schema = self._resolve_schema(
            schema_model=schema_model,
            response_schema=response_schema,
            schema_file=schema_file,
        )
        self._json_schema_str = (
            json.dumps(self.response_schema) if self.response_schema else None
        )

        if self._json_schema_str:
            source = (
                getattr(schema_model, "__name__", "custom")
                if schema_model
                else Path(schema_file).name
                if schema_file
                else "inline JSON"
            )
            eval_logger.info(f"Structured outputs enabled via schema '{source}'.")
        else:
            eval_logger.info("No schema provided; falling back to vanilla SGLangLM.")

        if self.use_remote_api:
            self._init_remote_backend(pretrained, kwargs)
        else:
            # Ensure device is set for local engine mode
            if "device" not in kwargs or kwargs["device"] is None:
                kwargs["device"] = "cuda"
            super().__init__(pretrained=pretrained, **kwargs)
        
    # -------------------------------------------------------------------------
    # Generation overrides
    # -------------------------------------------------------------------------
    def modify_gen_kwargs(self, kwargs: dict) -> dict:
        """Attach `json_schema` so Engine enforces `sgl.json(...)` semantics."""
        updated = super().modify_gen_kwargs(kwargs)
        if self._json_schema_str and "json_schema" not in updated:
            updated["json_schema"] = self._json_schema_str
        return updated

    def generate_until(
        self, requests, disable_tqdm: bool = False  # type: ignore[override]
    ):
        if self.use_remote_api:
            results = self._remote_generate_until(requests, disable_tqdm=disable_tqdm)
        else:
            results = super().generate_until(requests, disable_tqdm=disable_tqdm)
        if not self.validate_with_pydantic:
            return results
        return [self._validate_output(text) for text in results]

    # -------------------------------------------------------------------------
    # Remote backend helpers
    # -------------------------------------------------------------------------
    def _init_remote_backend(self, pretrained: str, kwargs: dict):
        from lm_eval.api.model import TemplateLM

        TemplateLM.__init__(self)
        self._rank = 0
        self._max_length = kwargs.get("max_model_len") or kwargs.get("context_length")
        self._max_gen_toks = kwargs.get("max_gen_toks", 256)
        self.add_bos_token = kwargs.get("add_bos_token", False)
        self.custom_prefix_token_id = kwargs.get("prefix_token_id")
        batch_size = kwargs.get("batch_size", 1)
        self.batch_size = (
            "auto"
            if isinstance(batch_size, str) and "auto" in batch_size
            else int(batch_size)
        )
        self.think_end_token = kwargs.get("think_end_token")
        self.model = None
        self.tokenizer = self._init_remote_tokenizer(
            pretrained, kwargs.get("trust_remote_code", True)
        )
        eval_logger.info(f"Using remote SGLang server at {self.base_url}")

    def _init_remote_tokenizer(
        self, pretrained: str, trust_remote_code: bool = True
    ):
        """Mirror RemoteTokenizer fallback sequence."""
        try:
            from lm_eval.utils import RemoteTokenizer, check_remote_tokenizer_support

            if check_remote_tokenizer_support(self.base_url):
                eval_logger.info("Using remote tokenizer from SGLang server.")
                return RemoteTokenizer(self.base_url)
        except Exception as exc:
            eval_logger.warning(
                f"Remote tokenizer unavailable ({exc}); falling back to HuggingFace."
            )

        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    @property
    def eot_token_id(self):
        if self.use_remote_api:
            if hasattr(self.tokenizer, "eos_token_id"):
                return self.tokenizer.eos_token_id
            if hasattr(self.tokenizer, "tokenizer_info"):
                return self.tokenizer.tokenizer_info.get("eos_token_id")
            return None
        return super().eot_token_id

    @property
    def prefix_token_id(self):
        if self.use_remote_api:
            if self.custom_prefix_token_id is not None:
                return self.custom_prefix_token_id
            if hasattr(self.tokenizer, "bos_token_id") and self.tokenizer.bos_token_id:
                return self.tokenizer.bos_token_id
            return self.eot_token_id
        return super().prefix_token_id

    @property
    def max_length(self):
        if self.use_remote_api:
            if self._max_length:
                return self._max_length
            if hasattr(self.tokenizer, "model_max_length"):
                return self.tokenizer.model_max_length
            if hasattr(self.tokenizer, "tokenizer_info"):
                return self.tokenizer.tokenizer_info.get("model_max_length", 2048)
            return 2048
        return super().max_length

    @property
    def max_gen_toks(self):
        if self.use_remote_api:
            return self._max_gen_toks
        return super().max_gen_toks

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ):
        if not self.use_remote_api:
            return super().tok_encode(
                string,
                left_truncate_len=left_truncate_len,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
            )

        if not add_special_tokens:
            add_special_tokens = self.add_bos_token

        if hasattr(self.tokenizer, "__call__"):
            encoding = self.tokenizer(
                string,
                add_special_tokens=add_special_tokens,
                truncation=truncation,
                return_attention_mask=False,
            ).input_ids
        else:
            if isinstance(string, str):
                encoding = self.tokenizer.encode(
                    string, add_special_tokens=add_special_tokens
                )
            else:
                encoding = [
                    self.tokenizer.encode(s, add_special_tokens=add_special_tokens)
                    for s in string
                ]

        if left_truncate_len:
            if isinstance(string, str):
                encoding = encoding[-left_truncate_len:]
            else:
                encoding = [enc[-left_truncate_len:] for enc in encoding]
        return encoding

    def tok_decode(self, tokens: List[int]) -> str:
        if self.use_remote_api and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(tokens)
        return super().tok_decode(tokens)

    def _remote_generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res: List[str] = []
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding = self.tok_encode(
            context, add_special_tokens=self.add_bos_token
        )
        requests_with_encoding = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]

        def _collate_gen(_requests):
            return -len(_requests[0][1]), _requests[0][0]

        re_ords = Collator(requests_with_encoding, _collate_gen, group_by=None)
        chunks = re_ords.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running schema-constrained generate_until requests",
        )
        eos = self.tokenizer.decode(self.eot_token_id) if self.eot_token_id else None

        for chunk in chunks:
            context_and_encoding, chunk_gen_kwargs = zip(*chunk)
            chunk_context, chunk_encoding = zip(*context_and_encoding)

            context_encoding_truncated = []
            sampling_params = []
            stop_sequences = []

            for tokens, gen_kwargs in zip(chunk_encoding, chunk_gen_kwargs):
                if not isinstance(gen_kwargs, dict):
                    raise ValueError(
                        f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                    )
                kwargs = copy.deepcopy(gen_kwargs)
                until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
                max_gen_toks = kwargs.pop("max_gen_toks", self.max_gen_toks)
                max_ctx_len = self.max_length - max_gen_toks

                if len(tokens) > max_ctx_len:
                    context_encoding_truncated.append(tokens[-max_ctx_len:])
                else:
                    context_encoding_truncated.append(tokens)

                kwargs = self.modify_gen_kwargs(kwargs)
                sampling_param = kwargs | {"max_tokens": max_gen_toks, "stop": until}
                sampling_params.append(sampling_param)
                stop_sequences.append(until)

            cont = self._model_generate(
                requests=context_encoding_truncated,
                generate=True,
                sampling_params=sampling_params,
            )

            for output, context_str, until, gen_kwargs in zip(
                cont, chunk_context, stop_sequences, chunk_gen_kwargs
            ):
                generated_text = output.get("text", "")
                generated_text = postprocess_generated_text(
                    generated_text, until, self.think_end_token
                )
                res.append(generated_text)
                self.cache_hook.add_partial(
                    "generate_until", (context_str, gen_kwargs), generated_text
                )
                pbar.update(1)

        pbar.close()
        return re_ords.get_original(res)

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        sampling_params: Union[List[Dict], Dict, None] = None,
        return_logprob: bool = False,
        top_logprobs_num: int = 1,
        logprob_start_len: int = -1,
    ):
        if not self.use_remote_api:
            return super()._model_generate(
                requests=requests,
                generate=generate,
                sampling_params=sampling_params,
                return_logprob=return_logprob,
                top_logprobs_num=top_logprobs_num,
                logprob_start_len=logprob_start_len,
            )

        import requests as http_requests

        if not generate:
            sampling_params = sampling_params if sampling_params else {}
            sampling_params.update({"temperature": 0, "max_new_tokens": 1})
        if not isinstance(sampling_params, List):
            sampling_params = [sampling_params] * len(requests)

        outputs = []
        for request_tokens, sp in zip(requests, sampling_params):
            payload = {"input_ids": request_tokens, "sampling_params": sp}
            if return_logprob:
                payload.update(
                    {
                        "return_logprob": True,
                        "top_logprobs_num": top_logprobs_num,
                        "logprob_start_len": logprob_start_len,
                    }
                )
            try:
                response = http_requests.post(
                    f"{self.base_url}/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=30,
                )
                response.raise_for_status()
                outputs.append(response.json())
            except Exception as exc:
                eval_logger.error(f"Remote SGLang request failed: {exc}")
                outputs.append(
                    {"text": "", "meta_info": {"input_token_logprobs": [], "input_top_logprobs": []}}
                )
        return outputs

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _resolve_schema(
        self,
        schema_model: Optional[type],
        response_schema: Optional[Union[Dict, str]],
        schema_file: Optional[str],
    ) -> Optional[Dict]:
        if schema_model:
            if BaseModel is None or not issubclass(schema_model, BaseModel):
                raise ValueError("schema_model must inherit from pydantic.BaseModel")
            return schema_model.model_json_schema()

        if schema_file:
            return self._load_schema_from_path(schema_file)

        if response_schema is None:
            return None
        
        if isinstance(response_schema, dict):
            return response_schema

        if isinstance(response_schema, str):
            path_candidate = Path(response_schema)
            if path_candidate.exists():
                return self._load_schema_from_path(path_candidate)
            try:
                return json.loads(response_schema)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"response_schema string is neither JSON nor a file path: {exc}"
                ) from exc

        raise ValueError("response_schema must be a dict, JSON string, or file path")

    @staticmethod
    def _load_schema_from_path(path_like: Union[str, Path]) -> Dict:
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Schema file {path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("Schema file must contain a JSON object.")
        return data

    def _validate_output(self, text: str) -> str:
        """One-shot validation: rely on SGLang to emit JSON, Pydantic to verify."""
        if not isinstance(text, str) or not text.strip():
            return text

        model_cls = self.schema_model
        if not model_cls or BaseModel is None:
            return text

        try:
            validated = model_cls.model_validate_json(text)
            return validated.model_dump_json()
        except ValidationError as exc:
            message = f"Pydantic validation failed: {exc}"
        except Exception as exc:  # pragma: no cover - defensive
            message = f"Unexpected validation error: {exc}"

        if self.strict_validation:
            raise ValueError(message)

        eval_logger.warning(message)
        return text

    # -------------------------------------------------------------------------
    # CLI helper
    # -------------------------------------------------------------------------
    @classmethod
    def create_from_arg_string(
        cls, arg_string: str, additional_config: Optional[dict] = None
    ):
        args = simple_parse_args_string(arg_string)
        if additional_config:
            args.update(additional_config)
        
        for key in ("validate_with_pydantic", "strict_validation"):
            if key in args and isinstance(args[key], str):
                args[key] = args[key].lower() in {"1", "true", "yes", "on"}

        return cls(**args)