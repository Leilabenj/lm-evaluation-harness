import copy
import io
import json
import logging
import os
import re
from contextlib import redirect_stdout
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
        Dynamically create a Pydantic model from a JSON Schema dictionary.
        
        Uses datamodel-code-generator library to convert JSON Schema to Pydantic v2 models.
        This handles complex cases like nested objects, arrays, enums, unions, etc.
        
        Args:
            schema_dict: JSON Schema dictionary
            
        Returns:
            Pydantic BaseModel class, or None if creation fails
            
        Note:
            If this approach encounters problems (library not available, generation errors, etc.),
            consider implementing a hybrid approach that falls back to manual schema-to-Pydantic
            mapping for basic schemas.
        """
        if BaseModel is None:
            eval_logger.error("Pydantic is not installed. Cannot create Pydantic model.")
            return None
        
        try:
            from datamodel_code_generator import InputFileType, generate
        except ImportError:
            eval_logger.error(
                "datamodel-code-generator not installed. "
                "Install with: pip install datamodel-code-generator"
            )
            return None
        
        try:
            # Convert schema dict to JSON string
            schema_json = json.dumps(schema_dict, indent=2)
            
            # Generate Pydantic v2 model code from JSON Schema
            # Note: generate() writes to stdout, so we need to capture it
            # Default output already generates Pydantic v2 compatible code
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                generate(
                    schema_json,
                    input_file_type=InputFileType.JsonSchema,
                    # Additional options for better compatibility
                    use_standard_collections=True,
                    use_schema_description=True,
                    use_field_description=True,
                )
            model_code = output_buffer.getvalue()
            
            if not model_code or not isinstance(model_code, str):
                eval_logger.error(
                    "Failed to generate model code from JSON Schema. "
                    "generate() did not produce output."
                )
                return None
            
            # Execute the generated code in a new namespace
            # This creates the model class(es) defined in the generated code
            # We need to include __builtins__ and ensure all necessary imports are available
            from typing import Optional, Union, List, Dict, Any, Tuple
            namespace = {'__builtins__': __builtins__}
            
            # Add typing imports that generated code commonly uses
            namespace.update({
                'Optional': Optional,
                'Union': Union,
                'List': List,
                'Dict': Dict,
                'Any': Any,
                'Tuple': Tuple,
            })
            
            # Add Pydantic imports to namespace so generated code can use them
            if BaseModel is not None:
                try:
                    from pydantic import confloat, conint, constr, conlist, conbytes, condate
                    namespace.update({
                        'BaseModel': BaseModel,
                        'confloat': confloat,
                        'conint': conint,
                        'constr': constr,
                        'conlist': conlist,
                        'conbytes': conbytes,
                        'condate': condate,
                    })
                except ImportError:
                    namespace['BaseModel'] = BaseModel
            
            exec(model_code, namespace)
            
            # Find the BaseModel class in the namespace
            # The generated model is typically named based on the schema's title,
            # or defaults to "Model" or "Root" if no title is provided
            model_class = None
            
            # Try common default names first
            for name in ['Model', 'Root', 'RootModel', 'Schema']:
                if name in namespace and isinstance(namespace[name], type) and issubclass(namespace[name], BaseModel):
                    model_class = namespace[name]
                    break
            
            # If not found, search for any BaseModel subclass
            if model_class is None:
                for name, obj in namespace.items():
                    if (isinstance(obj, type) and 
                        issubclass(obj, BaseModel) and 
                        obj is not BaseModel):
                        model_class = obj
                        break
            
            if model_class is None:
                eval_logger.error(
                    "Could not find generated Pydantic model class in generated code. "
                    "Generated code may not have created a model."
                )
                eval_logger.debug(f"Generated code namespace keys: {list(namespace.keys())}")
                return None
            
            # Call model_rebuild() for Pydantic v2 compatibility
            # This is required when using types like confloat, conint, etc. that have forward references
            # According to Pydantic docs: https://errors.pydantic.dev/2.12/u/class-not-fully-defined
            # The model needs to be rebuilt after all types (like confloat, Optional) are defined
            # Since we've included all necessary types in the namespace, model_rebuild() should work
            try:
                if hasattr(model_class, 'model_rebuild'):
                    # Force rebuild to resolve forward references (confloat, Optional, etc.)
                    model_class.model_rebuild(force=True)
            except Exception as e:
                eval_logger.warning(
                    f"Failed to rebuild Pydantic model (may still work): {e}"
                )
            
            eval_logger.debug(
                f"Successfully created Pydantic model '{model_class.__name__}' from JSON Schema"
            )
            return model_class
            
        except Exception as e:
            eval_logger.error(
                f"Failed to create Pydantic model from JSON Schema: {e}",
                exc_info=True
            )
            # Note: If we encounter persistent problems with this library-based approach,
            # we can implement a hybrid approach that:
            # 1. Tries this library method first
            # 2. Falls back to manual recursive schema-to-Pydantic mapping for basic schemas
            # 3. Handles common cases (objects, arrays, primitives, enums) manually
            # See the method docstring and comments for more details on fallback strategy.
            return None

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
        if not text or not isinstance(text, str):
            return None
        
        # Try custom regex pattern first if provided
        if self.json_pattern:
            try:
                match = re.search(self.json_pattern, text, re.DOTALL)
                if match:
                    extracted = match.group(1) if match.groups() else match.group(0)
                    # Validate it's valid JSON
                    try:
                        json.loads(extracted)
                        return extracted
                    except json.JSONDecodeError:
                        pass  # Continue to other methods
            except re.error as e:
                eval_logger.warning(f"Invalid custom JSON pattern: {e}")
        
        # Try extracting from markdown code blocks
        # Pattern: ```json\n{...}\n``` or ```\n{...}\n```
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
                # Validate it's valid JSON
                try:
                    json.loads(extracted)
                    return extracted
                except json.JSONDecodeError:
                    continue  # Try next pattern
        
        # Try finding JSON object/array in text using regex
        # This is more complex - we need to find balanced braces/brackets
        
        # Pattern for JSON object: { ... }
        # We'll try to find the first { and match it with the closing }
        json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        
        # Pattern for JSON array: [ ... ]
        json_array_pattern = r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]'
        
        # Try JSON object first
        for match in re.finditer(json_object_pattern, text, re.DOTALL):
            extracted = match.group(0)
            try:
                # Validate it's valid JSON
                json.loads(extracted)
                return extracted
            except json.JSONDecodeError:
                continue
        
        # Try JSON array
        for match in re.finditer(json_array_pattern, text, re.DOTALL):
            extracted = match.group(0)
            try:
                # Validate it's valid JSON
                json.loads(extracted)
                return extracted
            except json.JSONDecodeError:
                continue
        
        # More robust approach: find balanced braces/brackets
        # This handles nested structures better
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
                            # Found balanced JSON
                            extracted = text[start_idx:i+1]
                            try:
                                json.loads(extracted)
                                return extracted
                            except json.JSONDecodeError:
                                return None
            
            return None
        
        # Try finding JSON object with balanced braces
        json_obj = find_balanced_json(text, '{', '}')
        if json_obj:
            return json_obj
        
        # Try finding JSON array with balanced brackets
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
        Validate JSON string against Pydantic schema model.
        
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
            # Parse JSON string to Python dict
            try:
                parsed_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                return (False, None, f"Invalid JSON format: {str(e)}")
            
            # Instantiate Pydantic model with parsed data
            # This will validate the data against the schema
            validated_model = model_class(**parsed_data)
            
            # Success - return the validated model instance
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
            # Unexpected error during validation
            eval_logger.error(
                f"Unexpected error during Pydantic validation: {e}",
                exc_info=True
            )
            return (False, None, f"Unexpected validation error: {str(e)}")

    def _prepare_sglang_schema_params(self) -> Optional[dict]:
        """
        Prepare schema parameters for SGLang's structured output generation.
        
        SGLang supports structured outputs via its sampling parameters.
        This method converts our JSON Schema to SGLang's expected format.
        
        According to SGLang documentation:
        - JSON schema should be passed as a JSON string via 'json_schema' parameter
        - The schema is used to constrain model output during generation
        - Reference: https://docs.sglang.ai/advanced_features/structured_outputs.html
        
        Returns:
            Dictionary with SGLang schema parameters (e.g., {"json_schema": "..."}),
            or None if no schema is available
        """
        if not self.response_schema:
            return None
        
        try:
            # Convert JSON Schema dict to JSON string
            # SGLang expects the schema as a JSON string in the json_schema parameter
            json_schema_string = json.dumps(self.response_schema)
            
            # Return SGLang sampling parameters with json_schema
            # This will be merged into sampling_params in generate_until()
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
            # This ensures the output is properly formatted and validated
            try:
                # Use Pydantic's model_dump_json() for Pydantic v2, or json() for v1
                if hasattr(validated_model, 'model_dump_json'):
                    validated_json_str = validated_model.model_dump_json()
                elif hasattr(validated_model, 'json'):
                    validated_json_str = validated_model.json()
                else:
                    # Fallback: convert to dict and then to JSON
                    validated_json_str = json.dumps(validated_model.dict() if hasattr(validated_model, 'dict') else validated_model.__dict__)
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
                        # Success! Return validated JSON
                        if hasattr(retry_validated_model, 'model_dump_json'):
                            return retry_validated_model.model_dump_json()
                        elif hasattr(retry_validated_model, 'json'):
                            return retry_validated_model.json()
                        else:
                            return json.dumps(retry_validated_model.dict() if hasattr(retry_validated_model, 'dict') else retry_validated_model.__dict__)
                    
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