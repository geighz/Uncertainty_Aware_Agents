from UncertaintyAwareMiner import *


class UncertaintyAwareMinerNormalised(UncertaintyAwareMiner):
    # normalized variance
    def variance(self, predictions):
        predictions = torch.stack(predictions)
        var = predictions.var(dim=0)
        mean = abs(predictions.mean())
        return var / mean
