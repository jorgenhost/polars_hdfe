from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from polars_hdfe._internal import __version__ as __version__

if TYPE_CHECKING:
    from polars_hdfe.typing import IntoExprColumn

LIB = Path(__file__).parent

@pl.api.register_expr_namespace("hdfe_least_squares") 
class LeastSquaresNamespace:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def ols(self, features: list[IntoExprColumn], add_intercept: bool = True):
        if add_intercept:
            features = features + [pl.lit(1.0).alias("_intercept")]

        all_args = [self._expr] + features

        return register_plugin_function(
            args=all_args,
            plugin_path=LIB,
            function_name='pl_ols',
            is_elementwise=False
        )