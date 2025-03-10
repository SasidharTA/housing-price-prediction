# from featuretools.primitives import Second, Minute, Hour, Day, Week, Month, Year, Weekday, IsWeekend, \
# 	TimeSincePrevious, TimeSince
from .external import datetime_transformers as transformers

for agg in transformers:
    # exec('from featuretools.primitives import ' + agg.__name__)
    exec(f"{agg.__class__.__name__} = agg")
