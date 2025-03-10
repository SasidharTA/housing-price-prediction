# from featuretools.primitives import Absolute, Percentile, Diff, Negate, \
# 	CumCount, CumMax, CumMean, CumMin, CumSum, \
# 	AddNumeric, LessThanScalar, LessThanEqualTo, \
# 	Percentile

from .external import numeric_transformers as transformers

for agg in transformers:
    # exec('from featuretools.primitives import ' + agg.__name__)
    exec(f"{agg.__class__.__name__} = agg")
