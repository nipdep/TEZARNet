import numpy as np


def relative_weight_change(prev_list, curr_list):
    """Function computes the relative weight change (L1 Norm) between two lists

    Args:
        prev_list (list): NumPy list
        curr_list (list): NumPy list

    Returns:
        float: Relative Weight Change
    """

    # subtract
    sub = np.subtract(prev_list, curr_list)
    # absolute
    absolute = np.abs(sub)
    # normalization term
    normalizer = np.abs(prev_list)
    # relative error
    relative = absolute/normalizer

    met_dict = {'min': relative.min(), 'max': relative.max(), 'mean': relative.mean(), 'std': relative.std()}
    return met_dict


def setup_delta_tracking(model):
    """Function creates the initial dict to track relative weights and gets the random weights from the model

    Args:
        model (torch.Model): Torch model object
    Returns:
        weights (list): List of model weights
        layer_dict (dict): Empty dictionary of model layer weights
    """

    # load model layers.
    weights, layer_names = load_layers(model)

    # setup tracking dictionaries
    layers_dict = {}

    for layer in layer_names:
        layers_dict[layer] = []

    return weights, layers_dict


def compute_delta(model, prev_model_weights, rwc_deltas):
    """Function calculates Relative Weight Change between two model weights

    Args:
        model (torch.Model): Torch model object
        prev_model_weights (list): list of previous model weights
        rwc_deltas (dict): Dictionary to track the relative weight change in each layer

    Returns:
        rwc: updated delta dictionary of relative weights
    """

    # load model layers.
    curr_list, layer_names = load_layers(model)

    # update dictionaries with deltas
    for i, layer in zip(range(len(prev_model_weights)), layer_names):

        # compute RL1.
        layer_rmae_delta = relative_weight_change(
            prev_model_weights[i], curr_list[i])

        # update dictionaries.
        rwc_deltas[layer].append(layer_rmae_delta)

    # update previous.
    prev_list = curr_list.copy()

    return rwc_deltas, prev_list


def load_layers(model):
    layer_names, param_list = [], []

    for name, param in model.named_parameters():
        if len(param.size()) > 1:
            layer_names.append(name)
            param_list.append(param.clone().data.cpu().numpy())

    return param_list, layer_names