# Architecture: Schema-Constrained Model Backend

## Overview

This document breaks down the architecture of a model backend in lm-evaluation-harness into distinct layers. Understanding these layers is crucial before implementation.

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Interface Layer (LM base class)                    │
│ - Defines contract: loglikelihood, loglikelihood_rolling,   │
│   generate_until                                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Template Layer (TemplateLM - optional)             │
│ - Provides common tokenization helpers                       │
│ - Handles context/continuation encoding                      │
│ - Abstract methods: tok_encode, _loglikelihood_tokens        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Model Backend Layer (Your SchemaConstrainedLM)     │
│ - Initialization & configuration                            │
│ - Model-specific logic                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 4: Initialization Layer                                │
│ - Model loading (from_pretrained)                            │
│ - Tokenizer loading                                          │
│ - Device placement (CPU/GPU)                                 │
│ - Schema setup (Pydantic models)                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 5: Tokenization Layer                                  │
│ - Text → Token IDs (tok_encode)                             │
│ - Token IDs → Text (tok_decode)                              │
│ - Context/continuation pair encoding                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 6: Inference Layer                                      │
│ - Forward pass (_model_call)                                 │
│ - Logits computation                                         │
│ - Generation (_model_generate)                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 7: Post-Processing Layer (Schema-Specific)             │
│ - JSON extraction from generated text                        │
│ - Schema validation (Pydantic)                                │
│ - Error handling & retry logic                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 8: Caching Layer (CacheHook)                           │
│ - Request caching for efficiency                             │
│ - Cache key generation                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Interface Layer (`lm_eval.api.model.LM`)

**Purpose**: Defines the abstract contract that all model backends must implement.

**Key Components**:
- `loglikelihood(requests)` → `list[tuple[float, bool]]`
  - Computes log probability of continuation given context
  - Returns (logprob, is_greedy) for each request
  
- `loglikelihood_rolling(requests)` → `list[float]`
  - Computes full loglikelihood of a string (for perplexity)
  - Returns log probability for each request
  
- `generate_until(requests)` → `list[str]`
  - Generates text until stopping criteria
  - Returns generated text for each request

**Responsibilities**:
- Enforce consistent interface across all model types
- Define input/output types (Instance objects)
- Provide caching infrastructure (CacheHook)

**Your Implementation**: Must implement all three abstract methods.

---

## Layer 2: Template Layer (`lm_eval.api.model.TemplateLM`)

**Purpose**: Provides common functionality shared by tokenizer-based models.

**Key Components**:
- `tok_encode(string)` → `list[int]`
  - Tokenize text to token IDs
  
- `tok_decode(tokens)` → `str`
  - Decode token IDs to text
  
- `_encode_pair(context, continuation)` → `tuple[list[int], list[int]]`
  - Handles edge cases (spaces, special tokens)
  - Returns (context_tokens, continuation_tokens)
  
- `loglikelihood()` (concrete implementation)
  - Calls `_encode_pair()` then delegates to `_loglikelihood_tokens()`

**Responsibilities**:
- Abstract away tokenization boilerplate
- Handle context/continuation encoding edge cases
- Provide default implementations for common patterns

**Your Decision**: 
- **Option A**: Subclass `TemplateLM` (recommended if using tokenizers)
- **Option B**: Subclass `LM` directly (if you need full control)

**For Schema-Constrained**: Recommend `TemplateLM` since we'll use HuggingFace tokenizers.

---

## Layer 3: Model Backend Layer (Your `SchemaConstrainedLM`)

**Purpose**: Your specific implementation that adds schema validation.

**Key Components**:
- `__init__()` - Initialization with schema support
- `create_from_arg_string()` - Factory method for CLI instantiation
- Schema storage (Pydantic models, JSON Schema)
- Schema validation logic

**Responsibilities**:
- Coordinate all layers below
- Add schema-specific functionality
- Handle schema validation in `generate_until()`

**Schema-Specific Additions**:
- `response_schema`: JSON Schema dict or file path
- `schema_model`: Pydantic model class (optional)
- `_extract_json()`: Extract JSON from generated text
- `_validate_schema()`: Validate against Pydantic model

---

## Layer 4: Initialization Layer

**Purpose**: Load and configure the model, tokenizer, and schema.

**Key Components**:

### 4.1 Model Loading
```python
self._model = AutoModelForCausalLM.from_pretrained(
    pretrained,
    dtype=torch.bfloat16,
    device_map=device_map,
    trust_remote_code=trust_remote_code
)
```

### 4.2 Tokenizer Loading
```python
self.tokenizer = AutoTokenizer.from_pretrained(
    pretrained,
    use_fast=use_fast_tokenizer,
    trust_remote_code=trust_remote_code
)
```

### 4.3 Device Placement
```python
self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
self._model = self._model.to(self._device)
```

### 4.4 Schema Setup (NEW - Your Addition)
```python
# Load JSON Schema
if response_schema:
    if isinstance(response_schema, str):
        # Load from file
        with open(response_schema) as f:
            schema_dict = json.load(f)
    else:
        schema_dict = response_schema
    
    # Create Pydantic model from JSON Schema
    self.schema_model = self._create_pydantic_model(schema_dict)
else:
    self.schema_model = None
```

**Responsibilities**:
- Load model weights from HuggingFace or local path
- Load tokenizer (may be separate from model)
- Configure device (CPU/GPU)
- Set up schema validation infrastructure

---

## Layer 5: Tokenization Layer

**Purpose**: Convert between text and token representations.

**Key Components**:

### 5.1 Encoding (Text → Tokens)
```python
def tok_encode(self, string: str) -> list[int]:
    return self.tokenizer.encode(string, add_special_tokens=False)
```

### 5.2 Decoding (Tokens → Text)
```python
def tok_decode(self, tokens: list[int]) -> str:
    return self.tokenizer.decode(tokens, skip_special_tokens=True)
```

### 5.3 Pair Encoding (Context + Continuation)
- Handles edge cases (spaces, special tokens)
- Ensures continuation tokens align correctly
- Critical for accurate loglikelihood computation

**Responsibilities**:
- Maintain tokenization consistency
- Handle special tokens (BOS, EOS, PAD)
- Preserve word boundaries correctly

**Flow Example**:
```
Input: context="Hello" continuation=" world"
→ Encode: context_tokens=[1234, 5678], cont_tokens=[9012]
→ Model sees: [1234, 5678, 9012]
→ Logits computed for positions predicting continuation tokens
```

---

## Layer 6: Inference Layer

**Purpose**: Execute model forward passes and generation.

**Key Components**:

### 6.1 Forward Pass (`_model_call`)
```python
def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        outputs = self.model(inps)  # Forward pass
        return outputs.logits  # [batch, seq_len, vocab_size]
```

**What happens**:
- Input: Token IDs `[batch, sequence_length]`
- Model processes tokens autoregressively
- Output: Logits `[batch, sequence_length, vocab_size]`
- Each position has probability distribution over vocabulary

### 6.2 Loglikelihood Computation (`_loglikelihood_tokens`)
```python
def _loglikelihood_tokens(self, requests) -> list[tuple[float, bool]]:
    # 1. Batch requests together
    # 2. Call _model_call() to get logits
    # 3. Extract logits for continuation positions
    # 4. Compute log probabilities using log_softmax
    # 5. Sum log probs for continuation tokens
    # 6. Check if continuation is greedy (argmax matches)
```

**Process**:
```
Context: "The cat sat"
Continuation: " on the mat"

1. Tokenize: ctx=[1,2,3], cont=[4,5,6,7]
2. Forward pass: input=[1,2,3,4,5,6,7]
3. Get logits: logits[0,3:7,:] (positions 3,4,5,6)
4. Extract: logits for tokens [4,5,6,7]
5. Compute: log_prob = sum(log_softmax(logits)[tokens])
6. Check greedy: argmax(logits) == [4,5,6,7]?
```

### 6.3 Generation (`_model_generate` or `generate_until`)
```python
def _model_generate(self, context, max_length, stop, **kwargs):
    return self.model.generate(
        input_ids=context,
        max_length=max_length,
        stopping_criteria=stop,
        **kwargs
    )
```

**Process**:
```
1. Start with context tokens
2. Loop until stopping criteria:
   a. Forward pass → get logits for next position
   b. Sample next token (greedy or sampling)
   c. Append token to sequence
   d. Check stopping criteria (max_length, stop sequences)
3. Return generated token sequence
```

**Responsibilities**:
- Execute efficient batched forward passes
- Handle variable-length sequences (padding)
- Manage generation parameters (temperature, top_k, etc.)
- Optimize memory usage

---

## Layer 7: Post-Processing Layer (Schema-Specific)

**Purpose**: Validate and format generated outputs according to schema.

**Key Components**:

### 7.1 JSON Extraction
```python
def _extract_json(self, text: str) -> str:
    """
    Extract JSON from generated text.
    Handles cases where JSON is wrapped in:
    - Markdown code blocks (```json ... ```)
    - Plain text with JSON embedded
    - Pure JSON strings
    """
    # Try to find JSON in markdown code blocks
    # Try to find JSON object/array
    # Return clean JSON string
```

### 7.2 Schema Validation
```python
def _validate_schema(self, json_str: str) -> str:
    """
    Validate JSON against Pydantic model.
    Returns validated JSON string or raises ValidationError.
    """
    try:
        validated = self.schema_model.parse_raw(json_str)
        return validated.json()  # Re-serialize to ensure compliance
    except ValidationError as e:
        # Handle validation errors
        # Option: retry generation, return error message, etc.
```

### 7.3 Integration in `generate_until()`
```python
def generate_until(self, requests):
    results = []
    for request in requests:
        context, gen_kwargs = request.args
        
        # Step 1: Generate text (Layer 6)
        generated = self._model_generate(context, **gen_kwargs)
        decoded = self.tok_decode(generated)
        
        # Step 2: Schema validation (Layer 7)
        if self.schema_model:
            try:
                json_str = self._extract_json(decoded)
                validated = self._validate_schema(json_str)
                results.append(validated)
            except (JSONDecodeError, ValidationError) as e:
                # Error handling strategy
                results.append(f"ERROR: {str(e)}")
        else:
            results.append(decoded)
    
    return results
```

**Responsibilities**:
- Extract structured data from free-form text
- Validate against schema constraints
- Handle validation errors gracefully
- Ensure output format compliance

**Error Handling Strategies**:
1. **Return Error Message**: Include error in output
2. **Retry Generation**: Regenerate with adjusted prompt
3. **Fallback**: Return raw text if validation fails
4. **Logging**: Record validation failures for analysis

---

## Layer 8: Caching Layer (`CacheHook`)

**Purpose**: Cache model responses to avoid redundant computation.

**Key Components**:
- `cache_hook.add_partial()`: Store result in cache
- Hash-based cache keys: `hash_args(method_name, request_args)`
- SQLite backend for persistence

**How It Works**:
```python
# Before calling model
cache_key = hash_args("loglikelihood", request.args)
if cache_key in cache_db:
    return cache_db[cache_key]  # Use cached result

# Call model
result = self._loglikelihood_tokens(request)

# Store in cache
cache_hook.add_partial("loglikelihood", request.args, result)
```

**Responsibilities**:
- Speed up repeated evaluations
- Persist results across runs
- Handle cache invalidation

**For Schema Models**: Cache keys should include schema info to avoid using wrong schema's cache.

---

## Data Flow: Complete Example

### Example: `generate_until()` with Schema

```
1. Request arrives:
   Instance(args=("What are symptoms of diabetes?", {"max_new_tokens": 100}))

2. Layer 5 (Tokenization):
   context_tokens = tok_encode("What are symptoms of diabetes?")
   → [1234, 5678, 9012, ...]

3. Layer 6 (Inference):
   generated_tokens = model.generate(context_tokens, max_length=100)
   → [3456, 7890, 1234, ...]
   
   decoded_text = tok_decode(generated_tokens)
   → "Here are the symptoms:\n1. Increased thirst\n2. Frequent urination..."

4. Layer 7 (Post-Processing):
   json_str = _extract_json(decoded_text)
   → '{"answer": "Increased thirst, frequent urination...", "confidence": 0.95}'
   
   validated = _validate_schema(json_str)
   → Validated Pydantic model instance
   
   return validated.json()
   → '{"answer": "...", "confidence": 0.95}' (schema-compliant)

5. Layer 8 (Caching):
   cache_hook.add_partial("generate_until", request.args, result)
```

---

## Key Design Decisions for Schema-Constrained Model

### Decision 1: Inheritance Hierarchy
```
LM (abstract)
  ↓
TemplateLM (optional helper)
  ↓
SchemaConstrainedLM (your implementation)
```

**Recommendation**: Use `TemplateLM` for tokenization helpers.

### Decision 2: Schema Storage
- **Option A**: Store as Pydantic model class
- **Option B**: Store as JSON Schema dict, create Pydantic on-demand
- **Option C**: Both (flexibility)

**Recommendation**: Support both for maximum flexibility.

### Decision 3: Validation Timing
- **Option A**: Validate after every generation (current plan)
- **Option B**: Guide generation with schema (requires model support)
- **Option C**: Post-process with retry logic

**Recommendation**: Start with Option A, add retry logic.

### Decision 4: Error Handling
- **Option A**: Return error string in results
- **Option B**: Raise exceptions
- **Option C**: Log errors, return best-effort result

**Recommendation**: Option A (return error string) for robustness.

---

## Implementation Order

1. **Layer 4** (Initialization) - Load model, tokenizer, set up schema
2. **Layer 5** (Tokenization) - Implement `tok_encode`, `tok_decode`
3. **Layer 6** (Inference) - Implement `_model_call`, `_loglikelihood_tokens`, `_model_generate`
4. **Layer 7** (Post-Processing) - Implement JSON extraction and validation
5. **Layer 3** (Backend) - Wire everything together in main methods
6. **Layer 8** (Caching) - Ensure cache keys include schema info

---

## Next Steps

1. Review this architecture document
2. Study `huggingface.py` implementation for reference
3. Start with Layer 4 (Initialization) - simplest, no dependencies
4. Build up layer by layer, testing each as you go

