from . import CodeGenConfig

class CodeGenRLConfig(CodeGenConfig):
    model_type = "codegen-rl"
    def __init__(self, is_reward_model=False, **kwargs):
        super(CodeGenRLConfig, self).__init__(**kwargs)
        self.is_reward_model = is_reward_model
        self.has_value_head = True
        self.rl_algorithm = None # "ppo"

        self.ppo_clip = None
        self.ppo_vf_coef = None
        self.ppo_entropy_coef = None
