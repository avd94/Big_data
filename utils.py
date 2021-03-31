import json
import os
import numpy as np
import re
import copy

class Encoder(json.JSONEncoder):
  """Custom encoder so that we can save special types in JSON."""

  def default(self, obj):
    if isinstance(obj, (np.float_, np.float32, np.float16, np.float64)):
      return float(obj)
    elif isinstance(obj,
                    (np.intc, np.intp, np.int_, np.int8, np.int16, np.int32,
                     np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
      return int(obj)
    elif isinstance(obj, np.ndarray):
      obj = obj.tolist()
    return json.JSONEncoder.default(self, obj)


def save_dict(config_path, dict_with_info):
  """Saves a dict to a JSON file.
  Args:
    config_path: String with path where to save the gin config.
    dict_with_info: Dictionary with keys and values which are safed as strings.
  """
  # Ensure that the folder exists.
  directory = os.path.dirname(config_path)
  if not os.path.isdir(directory):
    os.makedirs(directory)
  # Save the actual config.
  with open(config_path, "w") as f:
    json.dump(dict_with_info, f, cls=Encoder, indent=2)

def _get_model_file(model_path, model_name):
    return os.path.join(model_path, model_name + ".tar")

def _get_result_file(model_path, model_name):
  return os.path.join(model_path, model_name + "_results.json")


def namespaced_dict(base_dict=None, **named_dicts):
  """Fuses several named dictionaries into one dict by namespacing the keys.
  Example:
  >> base_dict = {"!": "!!"}
  >> numbers = {"1": "one"}
  >> chars = {"a": "A"}
  >> new_dict = namespaced_dict(base_dict, numbers=numbers, chars=chars)
  >> # new_dict = {"!": "!!", "numbers.1": "one", "chars.a": "A"}
  Args:
    base_dict: Base dictionary of which a deepcopy will be use to fuse the named
      dicts into. If set to None, an empty dict will be used.
    **named_dicts: Named dictionary of dictionaries that will be namespaced and
      fused into base_dict. All keys should be string as the new key of any
      value will be outer key + "." + inner key.
  Returns:
    Dictionary with aggregated items.
  """
  result = {} if base_dict is None else copy.deepcopy(base_dict)
  for outer_key, inner_dict in named_dicts.items():
    for inner_key, value in inner_dict.items():
      result["{}.{}".format(outer_key, inner_key)] = value
  return result


def aggregate_json_results(base_path):
  """Aggregates all the result files in a directory into a namespaced dict.
  Args:
    base_path: String with the directory containing JSON files that only contain
      dictionaries.
  Returns:
    Namespaced dictionary with the results.
  """
  result = {}
  compiled_pattern = re.compile(r"(.*)\.json")
  for filename in os.listdir(base_path):
    match = compiled_pattern.match(filename)
    if match:
      path = os.path.join(base_path, filename)
      with open(path, "r") as f:
        result[match.group(1)] = json.load(f)
  return namespaced_dict(**result)