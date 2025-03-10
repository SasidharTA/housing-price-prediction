from .external import categorical_transformers as transformers

for agg in transformers:
    exec(f"{agg.__class__.__name__} = agg")
