import numpy

def success_rate(pred, true):
    cnt = 0
    for i in range(pred.shape[0]):
        t = numpy.where(true[i] == 1) # true set
        ary = numpy.intersect1d(pred[i], t)
        if ary.size > 0:
            cnt += 1
    return cnt * 100 / pred.shape[0]
