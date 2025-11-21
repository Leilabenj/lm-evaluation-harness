import copy
import json
import logging
import os
import re
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
from lm_eval.utils import simple_parse_args_string

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
    SGLang model backend with simplified support for structured outputs using Pydantic.
    
    This class extends SGLangLM to add schema-constrained generation capabilities using
    SGLang's native structured output support and Pydantic's native methods.
    
    Key features:
    - Direct Pydantic BaseModel support for schema definition
    - Automatic JSON schema generation via model.model_json_schema()
    - Native validation using model.model_validate_json()
    - SGLang's structured output constraints during generation
    - Simplified JSON extraction with fallback patterns
    - Error handling and optional retry logic
    
    Recommended usage:
        from pydantic import BaseModel, Field
        
        class MySchema(BaseModel):
            name: str = Field(..., description="Name field")
            age: int = Field(..., ge=0, description="Age field")
        
        model = SGLangSchemaLM(
            pretrained="meta-llama/Llama-2-7b-chat-hf",
            schema_model=MySchema
        )
    """
    
    def __init__(
        self,
        pretrained: str,
        # Schema-related arguments - simplified to use Pydantic model directly
        schema_model: Optional[BaseModel] = None,  # Pydantic model class for validation and schema generation
        # Legacy support for JSON schema (will be converted to Pydantic if needed)
        response_schema: Optional[Union[dict, str]] = None,  # JSON Schema dict or path to JSON file
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
            schema_model: Pydantic BaseModel class for validation and schema generation (preferred)
            response_schema: JSON Schema dict or path to JSON Schema file (legacy support)
            schema_file: Path to JSON Schema file (legacy support)
            validate_with_pydantic: Enable Pydantic validation after generation
            retry_on_validation_error: Retry generation if validation fails
            max_retries: Maximum retry attempts
            error_on_validation_failure: Raise exception vs return error string
            extract_json_from_markdown: Extract JSON from ```json code blocks
            json_pattern: Custom regex for JSON extraction
            **kwargs: All other arguments from SGLangLM parent class
        """

        # Check if base_url is provided for remote API mode
        self.base_url = kwargs.pop('base_url', None)
        self.use_remote_api = self.base_url is not None
        
        if self.use_remote_api:
            # For remote API mode, we need to initialize the parent class differently
            # to avoid creating a local engine. We'll do a minimal initialization.
            
            # Initialize TemplateLM directly instead of SGLangLM to avoid engine creation
            from lm_eval.api.model import TemplateLM
            TemplateLM.__init__(self)
            
            # Set required attributes that SGLangLM would normally set
            self.think_end_token = kwargs.get('think_end_token', None)
            self._max_length = kwargs.get('max_model_len') or kwargs.get('context_length')
            self.tensor_parallel_size = int(kwargs.get('tp_size', 1))
            self.data_parallel_size = int(kwargs.get('dp_size', 1))
            self._max_gen_toks = kwargs.get('max_gen_toks', 256)
            self.add_bos_token = kwargs.get('add_bos_token', False)
            self.custom_prefix_token_id = kwargs.get('prefix_token_id', None)
            
            # Set batch size
            batch_size = kwargs.get('batch_size', 1)
            self.batch_size = (
                "auto"
                if isinstance(batch_size, str) and "auto" in batch_size
                else int(batch_size)
            )
            
            # For remote API mode, we don't need to create a model object
            # We'll use HTTP requests directly in _model_generate
            self.model = None  # Set to None to indicate remote mode
            eval_logger.info(f"Using remote SGLang server at {self.base_url}")
            
            # Initialize tokenizer for remote mode
            self._init_remote_tokenizer(pretrained, kwargs.get('trust_remote_code', True))
        else:
            # For local engine mode, proceed normally
            if "device" not in kwargs or kwargs["device"] is None:
                kwargs["device"] = "cuda"
            
            # Initialize parent SGLangLM class normally
            super().__init__(pretrained=pretrained, **kwargs)
        
        # Store configuration
        self.validate_with_pydantic = validate_with_pydantic and (BaseModel is not None)
        self.retry_on_validation_error = retry_on_validation_error
        self.max_retries = max_retries
        self.error_on_validation_failure = error_on_validation_failure
        self.extract_json_from_markdown = extract_json_from_markdown
        self.json_pattern = json_pattern
        
        # Simplified schema handling: prioritize Pydantic model
        self.schema_model = schema_model
        self.response_schema = None
        
        if self.schema_model:
            # Generate JSON schema from Pydantic model using native method
            if hasattr(self.schema_model, 'model_json_schema'):
                self.response_schema = self.schema_model.model_json_schema()
                eval_logger.info(f"Generated JSON schema from Pydantic model: {self.schema_model.__name__}")
            else:
                eval_logger.error(f"Provided schema_model does not have model_json_schema() method")
                self.schema_model = None
        elif response_schema or schema_file:
            # Legacy support: load JSON schema and try to create Pydantic model
            eval_logger.warning(
                "Using legacy JSON schema support. Consider using schema_model with Pydantic BaseModel instead."
            )
            self.response_schema = self._load_schema(response_schema, schema_file)
            if self.response_schema and self.validate_with_pydantic:
                self.schema_model = self._create_pydantic_model_from_schema(self.response_schema)
        
        # Log final configuration
        if self.schema_model:
            eval_logger.info(f"Schema-constrained generation enabled with Pydantic model: {self.schema_model.__name__}")
        elif self.response_schema:
            eval_logger.info(f"Schema-constrained generation enabled with JSON schema (no validation)")
        else:
            eval_logger.info("No schema provided - falling back to standard generation")

    def _init_remote_tokenizer(self, pretrained: str, trust_remote_code: bool = True):
        """Initialize tokenizer for remote API mode."""
        try:
            from lm_eval.utils import RemoteTokenizer, check_remote_tokenizer_support
            if check_remote_tokenizer_support(self.base_url):
                self.tokenizer = RemoteTokenizer(self.base_url)
                eval_logger.info("Using remote tokenizer from SGLang server")
            else:
                # Fall back to HuggingFace tokenizer
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
                eval_logger.info("Using local HuggingFace tokenizer as fallback")
        except Exception as e:
            eval_logger.warning(f"Failed to initialize remote tokenizer: {e}. Using HuggingFace tokenizer.")
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)

    @property
    def eot_token_id(self):
        """Override to handle remote tokenizer."""
        if self.use_remote_api:
            if hasattr(self.tokenizer, 'eos_token_id'):
                return self.tokenizer.eos_token_id
            elif hasattr(self.tokenizer, 'tokenizer_info'):
                # For RemoteTokenizer
                return self.tokenizer.tokenizer_info.get('eos_token_id')
            else:
                # Fallback
                return getattr(self.tokenizer, 'eos_token_id', None)
        else:
            return super().eot_token_id

    @property
    def prefix_token_id(self):
        """Override to handle remote mode."""
        if self.use_remote_api:
            if self.custom_prefix_token_id is not None:
                return self.custom_prefix_token_id
            if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
                return self.tokenizer.bos_token_id
            return self.eot_token_id
        else:
            return super().prefix_token_id

    @property
    def max_length(self):
        """Override to handle remote mode."""
        if self.use_remote_api:
            if self._max_length:
                return self._max_length
            # For remote mode, use a reasonable default or try to get from tokenizer
            if hasattr(self.tokenizer, 'model_max_length'):
                return self.tokenizer.model_max_length
            elif hasattr(self.tokenizer, 'tokenizer_info'):
                return self.tokenizer.tokenizer_info.get('model_max_length', 2048)
            else:
                return 2048
        else:
            return super().max_length

    @property
    def max_gen_toks(self):
        """Override to handle remote mode."""
        if self.use_remote_api:
            return self._max_gen_toks
        else:
            return super().max_gen_toks

    def tok_encode(
        self,
        string: Union[str, List[str]],
        left_truncate_len: int = None,
        add_special_tokens: bool = False,
        truncation: bool = False,
    ) -> Union[List[int], List[List[int]]]:
        """Override to handle remote tokenizer."""
        if self.use_remote_api:
            if not add_special_tokens:
                add_special_tokens = False or self.add_bos_token
            
            if hasattr(self.tokenizer, '__call__'):
                # For RemoteTokenizer or HuggingFace tokenizer
                encoding = self.tokenizer(
                    string,
                    add_special_tokens=add_special_tokens,
                    truncation=truncation,
                    return_attention_mask=False,
                ).input_ids
            else:
                # Fallback for other tokenizer types
                if isinstance(string, str):
                    encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
                else:
                    encoding = [self.tokenizer.encode(s, add_special_tokens=add_special_tokens) for s in string]

            # left-truncate the encoded context to be at most `left_truncate_len` tokens long
            if left_truncate_len:
                if not isinstance(string, str):
                    encoding = [enc[-left_truncate_len:] for enc in encoding]
                else:
                    encoding = encoding[-left_truncate_len:]

            return encoding
        else:
            return super().tok_encode(string, left_truncate_len, add_special_tokens, truncation)

    def tok_decode(self, tokens: List[int]) -> str:
        """Override to handle remote tokenizer."""
        if self.use_remote_api:
            if hasattr(self.tokenizer, 'decode'):
                return self.tokenizer.decode(tokens)
            else:
                # Fallback
                return str(tokens)
        else:
            return super().tok_decode(tokens)

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
            
        Raises:
            ValueError: If schema cannot be loaded or is invalid
            FileNotFoundError: If schema file path doesn't exist
        """
        # If neither is provided, return None
        if not response_schema and not schema_file:
            return None
        
        # Priority: schema_file > response_schema
        # If both are provided, schema_file takes precedence
        if schema_file:
            if not os.path.exists(schema_file):
                raise FileNotFoundError(f"Schema file not found: {schema_file}")
            
            try:
                with open(schema_file, 'r', encoding='utf-8') as f:
                    schema_dict = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in schema file {schema_file}: {e}")
            except Exception as e:
                raise ValueError(f"Error reading schema file {schema_file}: {e}")
        elif response_schema:
            # Handle dict input (return as-is)
            if isinstance(response_schema, dict):
                schema_dict = response_schema
            # Handle string input
            elif isinstance(response_schema, str):
                # First, try to parse as JSON string
                try:
                    schema_dict = json.loads(response_schema)
                except json.JSONDecodeError:
                    # If JSON parsing fails, try as file path
                    if os.path.exists(response_schema):
                        try:
                            with open(response_schema, 'r', encoding='utf-8') as f:
                                schema_dict = json.load(f)
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"Invalid JSON in schema file {response_schema}: {e}"
                            )
                        except Exception as e:
                            raise ValueError(
                                f"Error reading schema file {response_schema}: {e}"
                            )
                    else:
                        raise ValueError(
                            f"Could not parse '{response_schema}' as JSON and "
                            f"file path does not exist"
                        )
            else:
                raise ValueError(
                    f"response_schema must be dict, str, or None, got {type(response_schema)}"
                )
        else:
            return None
        
        # Basic validation: ensure it's a dict
        if not isinstance(schema_dict, dict):
            raise ValueError(
                f"Schema must be a dictionary/object, got {type(schema_dict)}"
            )
        
        # Basic JSON Schema validation: check for common schema properties
        # This is a lightweight check - full validation would require jsonschema library
        if not schema_dict:
            raise ValueError("Schema dictionary cannot be empty")
        
        # Log successful load
        eval_logger.debug(f"Loaded JSON Schema with keys: {list(schema_dict.keys())}")
        
        return schema_dict

    def _create_pydantic_model_from_schema(self, schema_dict: dict) -> Optional[BaseModel]:
        """
        Legacy method for creating Pydantic model from JSON Schema.
        
        Note: This method is deprecated in favor of using Pydantic models directly.
        Consider defining your schema as a Pydantic BaseModel class instead of JSON Schema.
        
        Args:
            schema_dict: JSON Schema dictionary
            
        Returns:
            None (not implemented in simplified version)
        """
        eval_logger.warning(
            "Legacy JSON Schema to Pydantic conversion is no longer supported. "
            "Please define your schema as a Pydantic BaseModel class directly. "
            "Example:\n"
            "from pydantic import BaseModel, Field\n"
            "class MySchema(BaseModel):\n"
            "    name: str = Field(..., description='Name field')\n"
            "    age: int = Field(..., ge=0, description='Age field')\n"
            "Then use: SGLangSchemaLM(schema_model=MySchema, ...)"
        )
        return None

    def _extract_json(self, text: str) -> Optional[str]:
        """
        Extract JSON string from model output.
        
        With SGLang's structured outputs, the model should return valid JSON directly.
        This method provides fallback extraction for cases where the output includes
        additional text or markdown formatting.
        
        Args:
            text: Raw model output text
            
        Returns:
            Extracted JSON string, or None if no valid JSON found
        """
        if not text or not isinstance(text, str):
            return None
        
        text = text.strip()
        
        # First, try to parse the text directly as JSON (most common case with structured outputs)
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
        
        # Try custom regex pattern if provided
        if self.json_pattern:
            try:
                match = re.search(self.json_pattern, text, re.DOTALL)
                if match:
                    extracted = match.group(1) if match.groups() else match.group(0)
                    try:
                        json.loads(extracted)
                        return extracted
                    except json.JSONDecodeError:
                        pass
            except re.error as e:
                eval_logger.warning(f"Invalid custom JSON pattern: {e}")
        
        # Try extracting from markdown code blocks (common fallback)
        markdown_patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json\n...\n```
            r'```\s*\n(.*?)\n```',      # ```\n...\n``` (generic)
            r'```json\s*(.*?)```',       # ```json...``` (no newlines)
            r'```\s*(.*?)```',           # ```...``` (generic, no newlines)
        ]
        
        for pattern in markdown_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                try:
                    json.loads(extracted)
                    return extracted
                except json.JSONDecodeError:
                    continue
        
        # Simplified balanced brace/bracket finder for JSON objects and arrays
        def find_balanced_json(text: str, start_char: str, end_char: str) -> Optional[str]:
            """Find balanced JSON starting with start_char and ending with end_char."""
            start_idx = text.find(start_char)
            if start_idx == -1:
                return None
            
            depth = 0
            in_string = False
            escape_next = False
            
            for i in range(start_idx, len(text)):
                char = text[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                
                if not in_string:
                    if char == start_char:
                        depth += 1
                    elif char == end_char:
                        depth -= 1
                        if depth == 0:
                            extracted = text[start_idx:i+1]
                            try:
                                json.loads(extracted)
                                return extracted
                            except json.JSONDecodeError:
                                return None
            
            return None
        
        # Try finding JSON object or array
        json_obj = find_balanced_json(text, '{', '}')
        if json_obj:
            return json_obj
        
        json_arr = find_balanced_json(text, '[', ']')
        if json_arr:
            return json_arr
        
        # No valid JSON found
        return None

    def _validate_schema(
        self, 
        json_str: str, 
        schema_model: Optional[BaseModel] = None
    ) -> Tuple[bool, Optional[BaseModel], Optional[str]]:
        """
        Validate JSON string against Pydantic schema model using native model_validate_json().
        
        Args:
            json_str: JSON string to validate
            schema_model: Pydantic model class (uses self.schema_model if None)
            
        Returns:
            Tuple of (is_valid, validated_model_instance, error_message)
        """
        if BaseModel is None or ValidationError is None:
            return (False, None, "Pydantic is not installed")
        
        if not json_str or not isinstance(json_str, str):
            return (False, None, f"Invalid JSON string input: {type(json_str)}")
        
        # Get the schema model to use
        model_class = schema_model if schema_model is not None else self.schema_model
        
        if model_class is None:
            return (False, None, "No Pydantic schema model provided for validation")
        
        if not issubclass(model_class, BaseModel):
            return (False, None, f"Provided schema_model is not a Pydantic BaseModel: {type(model_class)}")
        
        try:
            # Use Pydantic's native model_validate_json() method - much simpler!
            validated_model = model_class.model_validate_json(json_str)
            return (True, validated_model, None)
            
        except ValidationError as e:
            # Pydantic validation failed
            # Format error message nicely
            error_messages = []
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error.get("loc", []))
                msg = error.get("msg", "Validation error")
                error_type = error.get("type", "unknown")
                error_messages.append(f"{loc}: {msg} (type: {error_type})")
            
            error_msg = "; ".join(error_messages) if error_messages else str(e)
            return (False, None, f"Validation failed: {error_msg}")
            
        except Exception as e:
            # Unexpected error during validation (e.g., invalid JSON)
            eval_logger.error(
                f"Unexpected error during Pydantic validation: {e}",
                exc_info=True
            )
            return (False, None, f"Unexpected validation error: {str(e)}")

    def _prepare_sglang_schema_params(self) -> Optional[dict]:
        """
        Prepare schema parameters for SGLang's structured output generation.
        
        Uses the JSON schema generated from Pydantic model via model_json_schema().
        SGLang expects the schema as a JSON string in the 'json_schema' parameter.
        
        Returns:
            Dictionary with SGLang schema parameters (e.g., {"json_schema": "..."}),
            or None if no schema is available
        """
        if not self.response_schema:
            return None
        
        try:
            # Convert JSON Schema dict (generated from Pydantic) to JSON string
            # SGLang expects the schema as a JSON string in the json_schema parameter
            json_schema_string = json.dumps(self.response_schema)
            
            # Return SGLang sampling parameters with json_schema
            return {"json_schema": json_schema_string}
            
        except (TypeError, ValueError) as e:
            eval_logger.error(
                f"Failed to convert JSON Schema to string for SGLang: {e}",
                exc_info=True
            )
            return None

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
        gen_kwargs: dict,
        is_retry: bool = False
    ) -> str:
        """
        Process generated text with schema extraction and validation.
        
        Args:
            generated_text: Raw generated text from model
            context: Original context (for retry if needed)
            gen_kwargs: Generation kwargs (for retry if needed)
            is_retry: Internal flag to prevent nested retries (default: False)
            
        Returns:
            Validated JSON string or error message
        """
        # Check if we're already in a retry - if so, disable retries to prevent infinite recursion
        can_retry = self.retry_on_validation_error and not is_retry
        
        # Step 1: Extract JSON from generated text
        json_str = self._extract_json(generated_text)
        
        if json_str is None:
            # JSON extraction failed
            if self.error_on_validation_failure:
                raise ValueError(
                    f"Failed to extract JSON from model output: {generated_text[:100]}..."
                )
            else:
                eval_logger.warning(
                    f"Failed to extract JSON from model output. "
                    f"Returning original text. Output: {generated_text[:100]}..."
                )
                return generated_text  # Return original text if extraction fails
        
        # Step 2: If validation is disabled, return extracted JSON
        if not self.validate_with_pydantic:
            return json_str
        
        # Step 3: Validate with Pydantic
        is_valid, validated_model, error_msg = self._validate_schema(json_str)
        
        if is_valid:
            # Validation succeeded - convert validated model back to JSON string
            # Use Pydantic's native model_dump_json() method
            try:
                validated_json_str = validated_model.model_dump_json()
                return validated_json_str
            except Exception as e:
                eval_logger.warning(
                    f"Failed to convert validated model to JSON string: {e}. "
                    f"Returning original extracted JSON."
                )
                return json_str
        
        # Validation failed
        # Retry if enabled and we're not already in a retry (prevents infinite recursion)
        if can_retry:
            # Retry generation up to max_retries times
            for attempt in range(1, self.max_retries + 1):
                eval_logger.debug(
                    f"Validation failed (attempt {attempt}/{self.max_retries}): {error_msg}. "
                    f"Retrying generation..."
                )
                
                try:
                    # Create an Instance for retry and call generate_until()
                    # This reuses all the existing logic (tokenization, batching, schema params, etc.)
                    retry_instance = Instance(
                        request_type=None,  # Not needed for generation
                        args=(context, gen_kwargs),
                        idx=None  # Not needed for single retry
                    )
                    
                    # Temporarily disable retries to prevent infinite recursion
                    # (generate_until will call _process_with_schema, which would retry again)
                    original_retry_setting = self.retry_on_validation_error
                    self.retry_on_validation_error = False
                    
                    try:
                        # Call generate_until with disable_tqdm=True to avoid progress bar noise
                        retry_results = self.generate_until(
                            [retry_instance],
                            disable_tqdm=True
                        )
                    finally:
                        # Restore original retry setting
                        self.retry_on_validation_error = original_retry_setting
                    
                    if not retry_results or len(retry_results) == 0:
                        continue  # Try again
                    
                    # The result from generate_until is already processed by _process_with_schema
                    # But we need to validate it again to check if it's valid
                    retry_text = retry_results[0]
                    
                    # Extract and validate the retry result
                    retry_json_str = self._extract_json(retry_text)
                    if retry_json_str is None:
                        # If it's an error message, try again
                        if retry_text.startswith("Validation error:"):
                            continue
                        # Otherwise, it might be valid JSON already
                        retry_json_str = retry_text
                    
                    retry_is_valid, retry_validated_model, retry_error_msg = self._validate_schema(retry_json_str)
                    
                    if retry_is_valid:
                        # Success! Return validated JSON using native method
                        return retry_validated_model.model_dump_json()
                    
                    # Validation still failed, continue to next retry
                    error_msg = retry_error_msg
                    
                except Exception as e:
                    eval_logger.warning(
                        f"Error during retry attempt {attempt}: {e}. "
                        f"Continuing to next retry..."
                    )
                    continue
            
            # All retries exhausted
            eval_logger.error(
                f"All {self.max_retries} retry attempts failed. "
                f"Last error: {error_msg}"
            )
        
        # No retries or retries exhausted - handle failure
        if self.error_on_validation_failure:
            raise ValueError(
                f"Schema validation failed after retries: {error_msg}. "
                f"Generated text: {generated_text[:200]}..."
            )
        else:
            # Return error message as string
            eval_logger.warning(
                f"Schema validation failed: {error_msg}. "
                f"Returning error message."
            )
            return f"Validation error: {error_msg}"

    def _model_generate(
        self,
        requests: List[List[int]] = None,
        generate: bool = False,
        sampling_params: Union[List[Dict], Dict, None] = None,
        return_logprob: bool = False,
        top_logprobs_num: int = 1,
        logprob_start_len: int = -1,
    ):
        """Override _model_generate to handle both local and remote modes."""
        if self.use_remote_api:
            # For remote API mode, use HTTP requests like sglang-generate model
            import requests as http_requests
            
            if not generate:
                sampling_params = sampling_params if sampling_params else {}
                sampling_params.update({
                    "temperature": 0,
                    "max_new_tokens": 1,
                })
            if not isinstance(sampling_params, List):
                sampling_params = [sampling_params] * len(requests)
            
            outputs = []
            for request_tokens, sp in zip(requests, sampling_params):
                # Create payload for SGLang server
                payload = {
                    "input_ids": request_tokens,
                    "sampling_params": sp,
                }
                
                if return_logprob:
                    payload.update({
                        "return_logprob": True,
                        "top_logprobs_num": top_logprobs_num,
                        "logprob_start_len": logprob_start_len,
                    })
                
                try:
                    # Make HTTP request to SGLang server
                    response = http_requests.post(
                        f"{self.base_url}/generate",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=30
                    )
                    response.raise_for_status()
                    output = response.json()
                    outputs.append(output)
                except Exception as e:
                    eval_logger.error(f"Failed to make request to SGLang server: {e}")
                    # Return a dummy output to avoid breaking the pipeline
                    outputs.append({"text": "", "meta_info": {"input_token_logprobs": [], "input_top_logprobs": []}})
            
            return outputs
        else:
            # For local engine mode, use parent implementation
            return super()._model_generate(
                requests=requests,
                generate=generate,
                sampling_params=sampling_params,
                return_logprob=return_logprob,
                top_logprobs_num=top_logprobs_num,
                logprob_start_len=logprob_start_len,
            )

    @classmethod
    def create_from_arg_string(
        cls, arg_string: str, additional_config: Optional[dict] = None
    ):
        """
        Create SGLangSchemaLM instance from argument string.
        
        This method parses CLI arguments like:
        pretrained=OpenMeditron/Meditron3-8B,response_schema=/path/to/schema.json,validate_with_pydantic=True
        
        Args:
            arg_string: Comma-separated key=value pairs
            additional_config: Optional additional configuration dict
            
        Returns:
            SGLangSchemaLM instance
        """
        # Parse argument string into dictionary
        args = simple_parse_args_string(arg_string)
        
        # Merge with additional_config if provided
        if additional_config:
            args.update(additional_config)
        
        # Handle boolean string values (e.g., "True", "False", "true", "false")
        boolean_params = [
            "validate_with_pydantic",
            "retry_on_validation_error",
            "error_on_validation_failure",
            "extract_json_from_markdown",
        ]
        for param in boolean_params:
            if param in args:
                value = args[param]
                if isinstance(value, str):
                    args[param] = value.lower() in ("true", "1", "yes", "on")
        
        # Handle integer string values
        int_params = ["max_retries"]
        for param in int_params:
            if param in args:
                value = args[param]
                if isinstance(value, str):
                    try:
                        args[param] = int(value)
                    except ValueError:
                        eval_logger.warning(
                            f"Could not convert {param}={value} to integer. "
                            f"Using default value."
                        )
                        args.pop(param)
        
        # Handle response_schema - could be a file path or JSON string
        # The _load_schema method will handle both cases
        if "response_schema" in args:
            # Keep as-is, _load_schema will handle file paths and JSON strings
            pass
        
        # Handle schema_file - ensure it's a string
        if "schema_file" in args:
            if not isinstance(args["schema_file"], str):
                eval_logger.warning(
                    f"schema_file must be a string, got {type(args['schema_file'])}. "
                    f"Ignoring."
                )
                args.pop("schema_file")
        
        # Instantiate and return SGLangSchemaLM
        return cls(**args)