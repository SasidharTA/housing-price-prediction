import featuretools.nlp_primitives as nlp_primitives
import inspect
from nlp_primitives import PartOfSpeechCount

from .external import import_and_load
from .external import text_transformers as transformers

nlp_classes = [x[1] for x in inspect.getmembers(nlp_primitives, inspect.isclass)]
transformers += import_and_load(nlp_classes)

for agg in transformers:
    # exec('from featuretools.primitives import ' + agg.__name__)
    exec(f"{agg.__class__.__name__} = agg")
