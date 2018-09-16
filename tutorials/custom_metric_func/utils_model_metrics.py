class MapeMetric:
    def map(self, predicted, actual, weight, offset, model):
        return [weight * abs((actual[0] - predicted[0]) / actual[0]), weight]

    def reduce(self, left, right):
        return [left[0] + right[0], left[1] + right[1]]

    def metric(self, last):
        return last[0] / last[1]

class CostMatrixLossMetric:
    def map(self, predicted, actual, weight, offset, model):
        cost_tp = 0 # set prior to use
        cost_tn = 0 # set prior to use
        cost_fp = 1 # set prior to use
        cost_fn = 3 # set prior to use

        c1 = cost_tp + cost_tn - cost_fp - cost_fn
        c2 = cost_fn - cost_tn
        c3 = cost_fp - cost_tn
        c4 = cost_tn

        y = actual[0]
        p = predicted[2] # [class, p0, p1]
        return [weight * ((y * p * c1) + (y * c2) + (p * c3) + c4), weight]

    def reduce(self, left, right):
        return [left[0] + right[0], left[1] + right[1]]

    def metric(self, last):
        return last[0] / last[1]

class WeightedFalseNegativeLossMetric:
    def map(self, predicted, actual, weight, offset, model):
        cost_tp = 5000 # set prior to use
        cost_tn = 0 # do not change
        cost_fp = cost_tp # do not change
        cost_fn = weight # do not change

        # c1 = cost_tp + cost_tn - cost_fp - cost_fn
        # c2 = cost_fn - cost_tn
        # c3 = cost_fp - cost_tn
        # c4 = cost_tn
        # (y * p * c1) + (y * c2) + (p * c3) + c4

        y = actual[0]
        p = predicted[2] # [class, p0, p1]
        if y == 1:
            denom = cost_fn
        else:
            denom = cost_fp
        return [(y * (1 - p) * cost_fn) + (p * cost_fp), denom]

    def reduce(self, left, right):
        return [left[0] + right[0], left[1] + right[1]]

    def metric(self, last):
        return last[0] / last[1]
