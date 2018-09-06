class MapeMetric:
    def map(self, predicted, actual, weight, offset, model):
        return [weight * abs((actual[0] - predicted[0]) / actual[0]), weight]

    def reduce(self, left, right):
        return [left[0] + right[0], left[1] + right[1]]

    def metric(self, last):
        return last[0] / last[1]
