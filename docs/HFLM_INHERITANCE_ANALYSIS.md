# Analysis: Inheriting from HFLM for Schema-Constrained Model

## Executive Summary

**YES - We should inherit from HFLM instead of TemplateLM!**

HFLM already provides all the infrastructure we need:
- ✅ Model loading and initialization
- ✅ Tokenizer setup
- ✅ Device management (CUDA/CPU/multi-GPU)
- ✅ All inference methods (`_loglikelihood_tokens`, `loglikelihood_rolling`, `generate_until`)
- ✅ Tokenization methods (`tok_encode`, `tok_decode`, `eot_token_id`)
- ✅ Batching, caching, and optimization

**We only need to:**
1. Override `__init__()` to add schema setup
2. Override `generate_until()` to add schema validation
3. Everything else works automatically!

---

## What HFLM Provides

### 1. Model Loading & Initialization (`__init__`)

**Methods:**
- `_get_config()` - Loads model configuration
- `_get_backend()` - Detects causal vs seq2seq
- `_create_model()` - Loads model with all options:
  - Quantization (8-bit, 4-bit, GPTQ, etc.)
  - PEFT/LoRA adapters
  - Multi-GPU support (device_map, parallelize)
  - Delta weights
  - GGUF format
- `_create_tokenizer()` - Loads tokenizer with all options

**Device Management:**
- Automatic device detection (CUDA/CPU/MPS/NPU/XPU)
- Multi-GPU support via Accelerate
- Model parallelism support
- Memory management

**Configuration:**
- Batch size (auto, fixed, per-GPU)
- Max length detection
- Dtype management (float16, bfloat16, etc.)
- Softmax dtype for numerical stability

### 2. Tokenization Methods

**Already Implemented:**
- `tok_encode(string)` → `list[int]` - Tokenizes text
- `tok_decode(tokens)` → `str` - Decodes tokens
- `eot_token_id` (property) - Returns EOS token ID
- `prefix_token_id` (property) - Returns BOS/EOS for prefix
- `tok_batch_encode()` - Batch tokenization

### 3. Inference Methods

**Already Implemented:**
- `_model_call(inps)` → `torch.Tensor` - Forward pass, returns logits
- `_model_generate(context, ...)` → `torch.Tensor` - Generation, returns token IDs
- `_loglikelihood_tokens(requests)` → `list[tuple[float, bool]]` - Full implementation
- `loglikelihood_rolling(requests)` → `list[float]` - Full implementation
- `generate_until(requests)` → `list[str]` - Full implementation

**Features:**
- Efficient batching with `Collator`
- Automatic batch size detection
- Caching support
- Multi-GPU support
- Stop sequence handling
- Truncation handling
- Padding management

### 4. Properties & Utilities

**Properties:**
- `self.model` - The actual model (with Accelerate unwrapping)
- `self.tokenizer` - The tokenizer
- `self.device` - Current device
- `self.max_length` - Max sequence length
- `self.batch_size` - Batch size
- `self.config` - Model config

**Utilities:**
- `_select_cont_toks()` - Extracts continuation logits
- `_detect_batch_size()` - Auto-detects optimal batch size
- `apply_chat_template()` - Applies chat templates

---

## What We Need to Add

### 1. Schema Setup in `__init__()`

```python
def __init__(self, response_schema=None, schema_model=None, **kwargs):
    # Call parent __init__ with all HFLM arguments
    super().__init__(**kwargs)
    
    # Add schema setup
    if response_schema:
        # Load JSON Schema from file or dict
        # Create Pydantic model
        self.schema_model = self._create_pydantic_model(response_schema)
    else:
        self.schema_model = schema_model  # Or None
```

### 2. Schema Validation in `generate_until()`

```python
def generate_until(self, requests, disable_tqdm=False):
    # Call parent method to get generated text
    results = super().generate_until(requests, disable_tqdm)
    
    # Add schema validation
    if self.schema_model:
        validated_results = []
        for generated_text in results:
            try:
                json_str = self._extract_json(generated_text)
                validated = self.schema_model.parse_raw(json_str)
                validated_results.append(validated.json())
            except (JSONDecodeError, ValidationError) as e:
                validated_results.append(f"ERROR: {str(e)}")
        return validated_results
    
    return results
```

### 3. Helper Methods

```python
def _extract_json(self, text: str) -> str:
    """Extract JSON from generated text (handles markdown, etc.)"""
    # Implementation here

def _create_pydantic_model(self, schema: dict | str) -> BaseModel:
    """Create Pydantic model from JSON Schema"""
    # Implementation here
```

---

## Comparison: TemplateLM vs HFLM

| Feature | TemplateLM | HFLM |
|---------|-----------|------|
| Model Loading | ❌ None | ✅ Complete |
| Tokenizer Setup | ❌ None | ✅ Complete |
| Device Management | ❌ None | ✅ Complete |
| `tok_encode()` | ⚠️ Abstract | ✅ Implemented |
| `tok_decode()` | ⚠️ Abstract | ✅ Implemented |
| `eot_token_id` | ⚠️ Abstract | ✅ Implemented |
| `_loglikelihood_tokens()` | ⚠️ Abstract | ✅ Implemented |
| `loglikelihood_rolling()` | ⚠️ Abstract | ✅ Implemented |
| `generate_until()` | ⚠️ Abstract | ✅ Implemented |
| `_model_call()` | ❌ None | ✅ Implemented |
| `_model_generate()` | ❌ None | ✅ Implemented |
| Batching | ❌ None | ✅ Complete |
| Caching | ✅ Provided | ✅ Provided |
| Multi-GPU | ❌ None | ✅ Supported |

**Verdict:** HFLM provides ~1500 lines of tested, production-ready code that we'd have to reimplement from TemplateLM.

---

## Implementation Strategy

### Option A: Inherit from HFLM (RECOMMENDED)

```python
@register_model("schema_constrained", "schema-constrained-llm")
class SchemaConstrainedLM(HFLM):
    def __init__(self, response_schema=None, schema_model=None, **kwargs):
        super().__init__(**kwargs)  # Gets all HFLM functionality
        # Add schema setup
        self.schema_model = self._setup_schema(response_schema, schema_model)
    
    def generate_until(self, requests, disable_tqdm=False):
        # Get generated text from parent
        results = super().generate_until(requests, disable_tqdm)
        
        # Add schema validation
        if self.schema_model:
            return [self._validate_and_format(r) for r in results]
        return results
    
    def _setup_schema(self, response_schema, schema_model):
        # Load/create Pydantic model
        pass
    
    def _validate_and_format(self, text):
        # Extract JSON and validate
        pass
```

**Pros:**
- ✅ Minimal code (~100 lines vs ~1500 lines)
- ✅ All HFLM features work automatically
- ✅ Tested, production-ready infrastructure
- ✅ Supports all HFLM options (quantization, PEFT, etc.)
- ✅ Multi-GPU support out of the box

**Cons:**
- ⚠️ Coupled to HFLM (but that's fine - we're using HF models anyway)

### Option B: Inherit from TemplateLM (NOT RECOMMENDED)

```python
@register_model("schema_constrained", "schema-constrained-llm")
class SchemaConstrainedLM(TemplateLM):
    def __init__(self, pretrained, response_schema=None, ...):
        # Reimplement ALL of HFLM's __init__ logic
        # Reimplement model loading
        # Reimplement tokenizer setup
        # Reimplement device management
        # Reimplement _loglikelihood_tokens
        # Reimplement loglikelihood_rolling
        # Reimplement generate_until
        # Add schema validation
        pass
```

**Pros:**
- ✅ More flexible (not tied to HF)

**Cons:**
- ❌ ~1500 lines of code to reimplement
- ❌ Need to test all edge cases
- ❌ Miss out on HFLM optimizations
- ❌ No multi-GPU support without extra work

---

## Code Changes Required

### 1. Update Imports

```python
from lm_eval.models.huggingface import HFLM
```

### 2. Change Inheritance

```python
class SchemaConstrainedLM(HFLM):  # Instead of TemplateLM
```

### 3. Override `__init__()`

```python
def __init__(self, response_schema=None, schema_model=None, **kwargs):
    super().__init__(**kwargs)
    # Add schema setup
```

### 4. Override `generate_until()`

```python
def generate_until(self, requests, disable_tqdm=False):
    results = super().generate_until(requests, disable_tqdm)
    # Add schema validation
    return validated_results
```

### 5. Add Helper Methods

- `_extract_json()` - Extract JSON from text
- `_create_pydantic_model()` - Create Pydantic model from JSON Schema
- `_validate_schema()` - Validate JSON against schema

---

## Example Usage

```bash
# CLI usage - all HFLM arguments work!
lm_eval --model schema_constrained \
    --model_args "pretrained=OpenMeditron/Meditron3-8B,response_schema=schemas/medical.json,device=cuda,batch_size=4" \
    --tasks hellaswag
```

All HFLM features work:
- ✅ Quantization: `autogptq=True`
- ✅ PEFT: `peft=/path/to/adapter`
- ✅ Multi-GPU: `parallelize=True`
- ✅ Custom tokenizer: `tokenizer=custom/tokenizer`
- ✅ And more!

---

## Conclusion

**Inherit from HFLM!** It's the right choice because:

1. **Minimal code**: ~100 lines vs ~1500 lines
2. **All features work**: Multi-GPU, quantization, PEFT, etc.
3. **Production-ready**: Tested, optimized, maintained
4. **Easy to maintain**: Changes to HFLM automatically benefit us
5. **Schema is the only addition**: We're just adding validation on top

The only thing we're adding is schema validation - everything else is already done!

