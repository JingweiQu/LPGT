import ast
import collections
import json
import os
import sys
import time
from collections.abc import Mapping
from copy import deepcopy
from warnings import warn
import shutil

import torch

JSON_FILE_KEY = 'default_json'


def delete_folder_contents_except(folder_path, excluded_folder):
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                if item in excluded_folder:
                    continue
                shutil.rmtree(item_path)

def unnormalize_points(points_norm, im_size):
    # normalize by image size
    w, h = im_size[0], im_size[1]
    points = points_norm.clone()
    points[:, 0] *= w
    points[:, 1] *= h

    return points


def PCK(p_src, p_wrp, L_pck, alpha=0.05):
    point_distance = torch.pow(torch.sum(torch.pow(p_src - p_wrp, 2), 1), 0.5)
    L_pck_mat = L_pck.expand_as(point_distance)
    correct_points = torch.le(point_distance, L_pck_mat * alpha)
    pck = torch.mean(correct_points.float())
    return pck


def compute_pck(graphs, im_sizes, L_pcks):
    pck = []
    alpha = 0.05
    for i in range(len(graphs)):
        points_norm_pred = graphs[i].x + graphs[i].y[:, 2:4]
        points_norm_gt = graphs[i].y[:, :2] + graphs[i].y[:, 2:4]
        im_size = im_sizes[i]
        points_pred = unnormalize_points(points_norm_pred, im_size)
        points_gt = unnormalize_points(points_norm_gt, im_size)
        pck.append(PCK(points_gt, points_pred, L_pcks[i], alpha).cpu().numpy())

    return pck


class ParamDict(dict):
    """ An immutable dict where elements can be accessed with a dot"""
    __getattr__ = dict.__getitem__

    def __delattr__(self, item):
        raise TypeError("Setting object not mutable after settings are fixed!")

    def __setattr__(self, key, value):
        raise TypeError("Setting object not mutable after settings are fixed!")

    def __setitem__(self, key, value):
        raise TypeError("Setting object not mutable after settings are fixed!")

    def __deepcopy__(self, memo):
        """ In order to support deepcopy"""
        return ParamDict([(deepcopy(k, memo), deepcopy(v, memo)) for k, v in self.items()])

    def __repr__(self):
        return json.dumps(self, indent=4, sort_keys=True)


def recursive_objectify(nested_dict):
    "Turns a nested_dict into a nested ParamDict"
    result = deepcopy(nested_dict)
    for k, v in result.items():
        if isinstance(v, Mapping):
            result[k] = recursive_objectify(v)
    return ParamDict(result)


class SafeDict(dict):
    """ A dict with prohibiting init from a list of pairs containing duplicates"""

    def __init__(self, *args, **kwargs):
        if args and args[0] and not isinstance(args[0], dict):
            keys, _ = zip(*args[0])
            duplicates = [item for item, count in collections.Counter(keys).items() if count > 1]
            if duplicates:
                raise TypeError("Keys {} repeated in json parsing".format(duplicates))
        super().__init__(*args, **kwargs)


def load_json(file):
    """ Safe load of a json file (doubled entries raise exception)"""
    with open(file, 'r') as f:
        data = json.load(f, object_pairs_hook=SafeDict)
    return data


def update_recursive(d, u, defensive=False):
    for k, v in u.items():
        if defensive and k not in d:
            raise KeyError("Updating a non-existing key")
        if isinstance(v, Mapping):
            d[k] = update_recursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def is_json_file(cmd_line):
    try:
        return os.path.isfile(cmd_line)
    except Exception as e:
        warn('JSON parsing suppressed exception: ', e)
        return False


def is_parseable_dict(cmd_line):
    try:
        res = ast.literal_eval(cmd_line)
        return isinstance(res, dict)
    except Exception as e:
        warn('Dict literal eval suppressed exception: ', e)
        return False


def update_params_from_cmdline(cmd_line=None, default_params=None, custom_parser=None, verbose=True):
    """ Updates default settings based on command line input.

    :param cmd_line: Expecting (same format as) sys.argv
    :param default_params: Dictionary of default params
    :param custom_parser: callable that returns a dict of params on success
    and None on failure (suppress exceptions!)
    :param verbose: Boolean to determine if final settings are pretty printed
    :return: Immutable nested dict with (deep) dot access. Priority: default_params < default_json < cmd_line
    """
    if not cmd_line:
        cmd_line = sys.argv

    if default_params is None:
        default_params = {}

    if len(cmd_line) < 2:
        cmd_params = {}
    elif custom_parser and custom_parser(cmd_line):  # Custom parsing, typically for flags
        cmd_params = custom_parser(cmd_line)
    elif len(cmd_line) == 2 and is_json_file(cmd_line[1]):
        cmd_params = load_json(cmd_line[1])
    elif len(cmd_line) == 2 and is_parseable_dict(cmd_line[1]):
        cmd_params = ast.literal_eval(cmd_line[1])
    else:
        raise ValueError('Failed to parse command line')

    update_recursive(default_params, cmd_params)

    if JSON_FILE_KEY in default_params:
        json_params = load_json(default_params[JSON_FILE_KEY])
        if 'default_json' in json_params:
            json_base = load_json(json_params[JSON_FILE_KEY])
        else:
            json_base = {}
        update_recursive(json_base, json_params)
        update_recursive(default_params, json_base)

    update_recursive(default_params, cmd_params)
    final_params = recursive_objectify(default_params)
    if verbose:
        print(final_params)

    update_params_from_cmdline.start_time = time.time()
    return final_params


update_params_from_cmdline.start_time = None
