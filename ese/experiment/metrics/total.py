# misc imports
from pydantic import validate_arguments
# ionpy imports
from ionpy.util import StatsMeter
from collections import defaultdict


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def global_ece_loss(
    pixel_preds: dict,
    ) -> dict:
    # Accumulate the dictionaries corresponding to a single bin.
    data_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (_, _, bin_num, measure), value in pixel_preds.items():
        data_dict[bin_num][measure].append(value)

    # Go through each bin and each measure (accurayc, weighted accuracy, confidence, weighted confidence).
    for bin_num, measures in data_dict.items():
        # Use metrics to keep track of stuff online.
        bin_metric_meters = {mg: StatsMeter() for mg in ["accuracy", "weighted accuracy", "confidence", "weighted confidence"]}
        # Loop through each x group and measure.
        for measure, values in measures.items():
            # Accumulate all of the values in this group, and at to our total bin trackers.
            for val in values:
                bin_metric_meters[measure] += val
