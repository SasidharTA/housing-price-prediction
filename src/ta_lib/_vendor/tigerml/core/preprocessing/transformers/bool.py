from .external import bool_transformers as transformers

for agg in transformers:
    # exec('from featuretools.primitives import ' + agg.__name__)
    exec(f"{agg.__class__.__name__} = agg")
