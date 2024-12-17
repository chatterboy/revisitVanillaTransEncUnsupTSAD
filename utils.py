class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# from Anoamly Transformer
def point_adjustment(gts, th_preds):
    anomaly_state = False

    for i in range(len(gts)):
        if gts[i] == 1 and th_preds[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gts[j] == 0:
                    break
                else:
                    if th_preds[j] == 0:
                        th_preds[j] = 1
            for j in range(i, len(gts)):
                if gts[j] == 0:
                    break
                else:
                    if th_preds[j] == 0:
                        th_preds[j] = 1
        elif gts[i] == 0:
            anomaly_state = False
        if anomaly_state:
            th_preds[i] = 1

    return gts, th_preds