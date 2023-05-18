from .codegen.configuration_codegen_rl import CodeGenRLConfig
from .codegen.modeling_flax_codegen_rl import FlaxCodeGenRLForCausalLM

# Register model
from transformers import AutoModelForCausalLM, FlaxAutoModelForCausalLM, AutoConfig
AutoConfig.register("codegen-rl", CodeGenRLConfig)
FlaxAutoModelForCausalLM.register(CodeGenRLConfig, FlaxCodeGenRLForCausalLM)
print(f"Model {FlaxCodeGenRLForCausalLM.__name__} registered as 'codegen-rl'")

