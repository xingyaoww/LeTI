import dataclasses
import flax.linen as nn
from flax.linen import partitioning as nn_partitioning
from typing import Mapping, Optional, Tuple

@dataclasses.dataclass
class ShardMixIn:
    """Adds parameter sharding constraints for any flax.linen Module.
    This is a mix-in class that overrides the `param` method of the
    original Module, to selectively add sharding constraints as specified
    in `shard_axes`"""

    shard_axes: Optional[Mapping[str, Tuple[str, ...]]] = None

    # Modifies off 
    # https://github.com/google/flax/blob/main/flax/linen/partitioning.py#L304
    def param(self, name: str, *init_args):
        # Initialize using the original Module's `param` method
        param = super().param(name, *init_args)

        # If `shard_axes` specified and param name in the dict, apply constraint
        if self.shard_axes and (name in self.shard_axes.keys()):
            axes = self.shard_axes[name]

            # Apply the sharding constraint (e.g. axes=('embedding', 'hidden'))
            param = nn_partitioning.with_sharding_constraint(param, axes)

            # Sow this, to have the AxisMetadata available at initialization.
            self.sow(
                "params_axes",
                f"{name}_axes",
                nn_partitioning.AxisMetadata(axes),
                reduce_fn=nn_partitioning._param_with_axes_sow_reduce_fn,
            )

        return param

class Dense(ShardMixIn, nn.Dense):
    pass

class Embed(ShardMixIn, nn.Embed):
    pass

class LayerNorm(ShardMixIn, nn.LayerNorm):
    pass
