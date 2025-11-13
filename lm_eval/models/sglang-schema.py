import copy
import json
import logging
import os
import re
from importlib.util import find_spec
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from lm_eval.models.sglang_causallms import SGLangLM
from lm_eval.models.utils import (
    Collator,
    handle_stop_sequences,
    postprocess_generated_text,
)

eval_logger = logging.getLogger(__name__)

try:
    import sglang as sgl
except ModuleNotFoundError:
    pass

try:
    from pydantic import BaseModel, ValidationError
    from pydantic.json import pydantic_encoder
except ImportError:
    BaseModel = None
    ValidationError = None
    eval_logger.warning(
        "Pydantic not installed. Schema validation will not work. "
        "Install with: pip install pydantic"
    )

if TYPE_CHECKING:
    pass


@register_model("sglang-schema")
class SGLangSchemaLM(SGLangLM):
    """
    SGLang model backend with support for structured outputs via JSON Schema.
    
    This class extends SGLangLM to add schema-constrained generation capabilities.
    It supports:
    - JSON Schema-based structured outputs via SGLang's native support
    - Pydantic model validation for post-generation validation
    - Automatic JSON extraction from model outputs (handles markdown code blocks, etc.)
    - Error handling for invalid schema responses
    """
    
    def __init__(
        self,
        pretrained: str,
        # Schema-related arguments
        response_schema: Optional[Union[dict, str]] = None,  # JSON Schema dict or path to JSON file
        schema_model: Optional[BaseModel] = None,  # Pydantic model class for validation
        schema_file: Optional[str] = None,  # Path to JSON Schema file
        # Schema validation options
        validate_with_pydantic: bool = True,  # Whether to validate with Pydantic after generation
        retry_on_validation_error: bool = False,  # Whether to retry generation on validation failure
        max_retries: int = 3,  # Maximum number of retries on validation failure
        error_on_validation_failure: bool = False,  # Whether to raise error or return error message
        # JSON extraction options
        extract_json_from_markdown: bool = True,  # Extract JSON from markdown code blocks
        json_pattern: Optional[str] = None,  # Custom regex pattern for JSON extraction
        # Inherit all SGLangLM arguments
        **kwargs,
    ):
        """
        Initialize SGLangSchemaLM with schema support.
        
        Args:
            pretrained: Model path or HuggingFace model identifier
            response_schema: JSON Schema dict or path to JSON Schema file
            schema_model: Pydantic BaseModel class for validation
            schema_file: Path to JSON Schema file (alternative to response_schema)
            validate_with_pydantic: Enable Pydantic validation after generation
            retry_on_validation_error: Retry generation if validation fails
            max_retries: Maximum retry attempts
            error_on_validation_failure: Raise exception vs return error string
            extract_json_from_markdown: Extract JSON from ```json code blocks
            json_pattern: Custom regex for JSON extraction
            **kwargs: All other arguments from SGLangLM parent class
        """
        # Initialize parent SGLangLM class
        super().__init__(pretrained=pretrained, **kwargs)
        
        # Load and store schema information
        self.response_schema = self._load_schema(response_schema, schema_file)
        self.schema_model = schema_model
        self.validate_with_pydantic = validate_with_pydantic and (BaseModel is not None)
        self.retry_on_validation_error = retry_on_validation_error
        self.max_retries = max_retries
        self.error_on_validation_failure = error_on_validation_failure
        self.extract_json_from_markdown = extract_json_from_markdown
        self.json_pattern = json_pattern
        
        # Create Pydantic model from JSON Schema if schema provided but no model
        if self.response_schema and not self.schema_model and self.validate_with_pydantic:
            self.schema_model = self._create_pydantic_model_from_schema(self.response_schema)
        
        # Log schema configuration
        if self.response_schema:
            eval_logger.info(f"Schema-constrained generation enabled with schema: {self.response_schema}")
        if self.schema_model:
            eval_logger.info(f"Pydantic validation enabled with model: {self.schema_model.__name__}")

    def _load_schema(
        self, 
        response_schema: Optional[Union[dict, str]], 
        schema_file: Optional[str]
    ) -> Optional[dict]:
        """
        Load JSON Schema from various sources.
        
        Args:
            response_schema: JSON Schema dict, JSON string, or file path
            schema_file: Path to JSON Schema file
            
        Returns:
            JSON Schema as dictionary, or None if not provided
        """
        # TODO: Implement schema loading logic
        # - Handle dict input (return as-is)
        # - Handle string input (try JSON parse, then try file path)
        # - Handle schema_file parameter
        # - Validate that loaded schema is valid JSON Schema
        pass

    def _create_pydantic_model_from_schema(self, schema_dict: dict) -> Optional[BaseModel]:
        """
        Dynamically create a Pydantic model from a JSON Schema dictionary.
        
        Args:
            schema_dict: JSON Schema dictionary
            
        Returns:
            Pydantic BaseModel class, or None if creation fails
        """
        # TODO: Implement dynamic Pydantic model creation
        # - Convert JSON Schema to Pydantic model
        # - Handle nested objects, arrays, enums, etc.
        # - Return model class that can be instantiated
        # Note: This is complex - consider using libraries like datamodel-code-generator
        # or manually mapping JSON Schema types to Pydantic types
        pass

    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON string from model output.
        
        Handles various formats:
        - Plain JSON: {"key": "value"}
        - Markdown code blocks: ```json\n{"key": "value"}\n```
        - Text with JSON: Some text {"key": "value"} more text
        - Custom patterns if json_pattern is provided
        
        Args:
            text: Raw model output text
            
        Returns:
            Extracted JSON string, or None if no valid JSON found
        """
        # TODO: Implement JSON extraction
        # - Try custom regex pattern if provided
        # - Try extracting from markdown code blocks (```json ... ```)
        # - Try finding JSON object/array in text using regex
        # - Validate extracted string is valid JSON
        # - Return first valid JSON found, or None
        pass

    def _validate_schema(
        self, 
        json_str: str, 
        schema_model: Optional[BaseModel] = None
    ) -> Tuple[bool, Optional[BaseModel], Optional[str]]:
        """
        Validate JSON string against Pydantic schema model.
        
        Args:
            json_str: JSON string to validate
            schema_model: Pydantic model class (uses self.schema_model if None)
            
        Returns:
            Tuple of (is_valid, validated_model_instance, error_message)
        """
        # TODO: Implement Pydantic validation
        # - Parse JSON string
        # - Instantiate Pydantic model with parsed data
        # - Catch ValidationError and return error details
        # - Return (True, model_instance, None) on success
        # - Return (False, None, error_msg) on failure
        pass

    def _prepare_sglang_schema_params(self) -> Optional[dict]:
        """
        Prepare schema parameters for SGLang's structured output generation.
        
        SGLang supports structured outputs via its sampling parameters.
        This method converts our JSON Schema to SGLang's expected format.
        
        Returns:
            Dictionary with SGLang schema parameters, or None if not using SGLang native support
        """
        # TODO: Implement SGLang schema parameter preparation
        # - Check SGLang documentation for structured output API
        # - Convert JSON Schema to SGLang's format
        # - Return parameters to pass to sampling_params
        # Reference: https://docs.sglang.ai/advanced_features/structured_outputs.html
        pass

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        """
        Generate text with optional schema constraints.
        
        This method extends the parent generate_until() to add:
        1. Schema-constrained generation via SGLang
        2. JSON extraction from outputs
        3. Pydantic validation
        4. Error handling and optional retries
        
        Args:
            requests: List of Instance objects with (context, gen_kwargs) args
            disable_tqdm: Whether to disable progress bar
            
        Returns:
            List of generated strings (validated JSON if schema enabled)
        """
        # If no schema is provided, fall back to parent implementation
        if not self.response_schema and not self.schema_model:
            return super().generate_until(requests, disable_tqdm=disable_tqdm)
        
        # TODO: Implement schema-constrained generation
        # 1. Prepare SGLang schema parameters if using native support
        # 2. Override/modify sampling_params to include schema constraints
        # 3. Call parent _model_generate() with schema parameters
        # 4. For each generated output:
        #    a. Extract JSON using _extract_json()
        #    b. Validate using _validate_schema() if enabled
        #    c. Handle validation errors (retry or return error message)
        #    d. Return validated JSON string or original text
        # 5. Maintain batching and caching behavior from parent class
        
        res = []
        
        # Batch tokenize contexts (similar to parent implementation)
        context, all_gen_kwargs = zip(*(req.args for req in requests))
        context_encoding: List[List[int]] = self.tok_encode(
            context, add_special_tokens=self.add_bos_token
        )
        requests_with_encoding = [
            ((a, b), c) for a, b, c in zip(context, context_encoding, all_gen_kwargs)
        ]
        
        # Collate and batch requests
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
        
        eos = self.tokenizer.decode(self.eot_token_id)
        
        # Prepare SGLang schema parameters if needed
        sglang_schema_params = self._prepare_sglang_schema_params()
        
        for chunk in chunks:
            context_and_encoding, all_gen_kwargs = zip(*chunk)
            context, context_encoding = zip(*context_and_encoding)
            
            context_encoding_truncated = []
            sampling_params = []
            
            for x, gen_kwargs in zip(context_encoding, all_gen_kwargs):
                # Unpack generation kwargs
                if isinstance(gen_kwargs, dict):
                    kwargs = copy.deepcopy(gen_kwargs)
                    until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
                else:
                    raise ValueError(
                        f"Expected `kwargs` to be of type `dict` but got {type(gen_kwargs)}"
                    )
                
                if "max_gen_toks" in kwargs.keys():
                    max_gen_toks = kwargs.pop("max_gen_toks")
                else:
                    max_gen_toks = self.max_gen_toks
                
                # Truncate context if needed
                max_ctx_len = self.max_length - max_gen_toks
                if len(x) > max_ctx_len:
                    context_encoding_truncated.append(x[-max_ctx_len:])
                else:
                    context_encoding_truncated.append(x)
                
                # Prepare sampling parameters
                kwargs = self.modify_gen_kwargs(kwargs)
                sampling_param = kwargs | {"max_tokens": max_gen_toks, "stop": until}
                
                # Add schema parameters if available
                if sglang_schema_params:
                    sampling_param.update(sglang_schema_params)
                
                sampling_params.append(sampling_param)
            
            # Perform batched generation
            cont = self._model_generate(
                requests=context_encoding_truncated,
                generate=True,
                sampling_params=sampling_params,
            )
            
            # Process outputs with schema validation
            for output, context_str, gen_kwargs in zip(cont, context, all_gen_kwargs):
                generated_text = output.get("text", "")
                generated_text = postprocess_generated_text(
                    generated_text, until, self.think_end_token
                )
                
                # Apply schema validation if enabled
                validated_text = self._process_with_schema(
                    generated_text, 
                    context_str, 
                    gen_kwargs
                )
                
                res.append(validated_text)
                self.cache_hook.add_partial(
                    "generate_until", (context_str, gen_kwargs), validated_text
                )
                pbar.update(1)
        
        pbar.close()
        return re_ords.get_original(res)

    def _process_with_schema(
        self, 
        generated_text: str, 
        context: str, 
        gen_kwargs: dict
    ) -> str:
        """
        Process generated text with schema extraction and validation.
        
        Args:
            generated_text: Raw generated text from model
            context: Original context (for retry if needed)
            gen_kwargs: Generation kwargs (for retry if needed)
            
        Returns:
            Validated JSON string or error message
        """
        # TODO: Implement schema processing pipeline
        # 1. Extract JSON from generated_text using _extract_json()
        # 2. If extraction fails, return error or original text
        # 3. If validate_with_pydantic:
        #    a. Validate extracted JSON using _validate_schema()
        #    b. If validation fails:
        #       - If retry_on_validation_error: retry generation (up to max_retries)
        #       - Else: return error message or raise exception
        #    c. If validation succeeds: return validated JSON string
        # 4. If no validation: return extracted JSON or original text
        pass

    @classmethod
    def create_from_arg_string(
        cls, arg_string: str, additional_config: Optional[dict] = None
    ):
        """
        Create SGLangSchemaLM instance from argument string.
        
        This method parses CLI arguments like:
        pretrained=OpenMeditron/Meditron3-8B,response_schema=/path/to/schema.json
        
        Args:
            arg_string: Comma-separated key=value pairs
            additional_config: Optional additional configuration dict
            
        Returns:
            SGLangSchemaLM instance
        """
        # TODO: Implement argument parsing
        # - Parse arg_string into dictionary
        # - Handle special cases (file paths, JSON strings, etc.)
        # - Merge with additional_config
        # - Instantiate and return SGLangSchemaLM
        # Reference: See parent class or other model implementations
        pass