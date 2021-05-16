import numpy as np

def quantile_loss(y_pred, target, quantiles=[0.025, 0.5, 0.975]):
    losses = []
    for i, q in enumerate(quantiles):
        error = target - y_pred[..., i]
        loss = np.max([(q - 1) * error, q * error], axis=0)
        losses.append(loss)
    losses = np.array(losses)
    return 2 * losses.sum() / np.abs(target + 1e-4).sum()