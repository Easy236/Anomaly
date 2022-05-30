import threshold
import numpy as np


def plot_prediction(model, data, device, title, ax):
    predictions, pred_losses = threshold.predict(model, [data], device)
    ax.plot(data, label='true')
    ax.plot(predictions[0], label='predicted')
    ax.set_title(f'{title} (loss: {np.around(pred_losses[0], 2)})')
    ax.legend()
