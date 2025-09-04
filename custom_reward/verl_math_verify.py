# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# try:
#     from math_verify.errors import TimeoutException
#     from math_verify.metric import math_metric
#     from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
#     from latex2sympy2_extended.latex2sympy2 import NormalizationConfig
# except ImportError:
#     print("To use Math-Verify, please install it first by running `pip install math-verify`.")


from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from latex2sympy2_extended.latex2sympy2 import NormalizationConfig

import logging
from contextlib import contextmanager

@contextmanager
def silence_logging(logger_name: str = "math_verify", level: int = logging.ERROR):
    """
    Temporarily raise the log level of `logger_name` to `level`.
    Everything under the `with` block is muted for messages below `level`.
    """
    logger = logging.getLogger(logger_name)
    prev_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(prev_level)


def compute_score(solution_str: str, ground_truth: str, data_source=None, extra_info=None) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(LatexExtractionConfig(boxed_match_priority=0, normalization_config=NormalizationConfig(basic_latex=True, units=True, malformed_operators=False, nits=False, boxed="all", equations=False)),),
    )
    ret_score = 0.0
    timeout_score = 0.0
    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    with silence_logging("math_verify", logging.ERROR):
        try:
            ret_score, _ = verify_func([ground_truth_boxed], [solution_str])
        except Exception:
            pass
        except TimeoutException:
            ret_score = timeout_score

    return ret_score