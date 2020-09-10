import math

import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


class WarmUpLinearCosineDecay(LearningRateSchedule):
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "num_periods": self.num_periods,
            "alpha": self.alpha,
            "beta": self.beta,
            "name": self.name
        }

    def __init__(
            self,
            initial_learning_rate,
            warmup_steps,
            decay_steps,
            num_periods=0.5,
            alpha=0.0,
            beta=0.001,
    ):
        super(WarmUpLinearCosineDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.num_periods = num_periods
        self.alpha = alpha
        self.beta = beta
        self.name = "WarmUpLinearCosineDecay"

    @tf.function
    def __call__(self, step):
        with ops.name_scope_v2(self.name or "LinearCosineDecay") as name:
            initial_learning_rate = ops.convert_to_tensor_v2(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            warmup_steps = math_ops.cast(self.warmup_steps, dtype)

            # warmup
            if step < warmup_steps:
                step = math_ops.cast(step, dtype)
                warmup = step / warmup_steps
                return math_ops.multiply(initial_learning_rate, warmup, name=name)
            else:
                step = math_ops.cast(step, dtype) - warmup_steps
                decay_steps = math_ops.cast(self.decay_steps, dtype) - warmup_steps
                num_periods = math_ops.cast(self.num_periods, dtype)
                alpha = math_ops.cast(self.alpha, dtype)
                beta = math_ops.cast(self.beta, dtype)

                global_step_recomp = math_ops.cast(step, dtype)
                global_step_recomp = math_ops.minimum(global_step_recomp, decay_steps)
                linear_decayed = (decay_steps - global_step_recomp) / decay_steps
                completed_fraction = global_step_recomp / decay_steps
                fraction = 2.0 * num_periods * completed_fraction
                cosine_decayed = 0.5 * (
                        1.0 + math_ops.cos(constant_op.constant(math.pi) * fraction))

                linear_cosine_decayed = (alpha + linear_decayed) * cosine_decayed + beta

                return math_ops.multiply(initial_learning_rate, linear_cosine_decayed,
                                         name=name)
