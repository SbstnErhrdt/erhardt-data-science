from typing import List, Any, Tuple

import numpy as np


def get_vector_as_x(data: List[Any]) -> Tuple[List[int], np.ndarray]:
    """
    Convert vector column to numpy array
    :param data:
    :return:
    """
    missing_data_index = []
    print('start converting embeddings to numpy array')
    result = []
    for i, x in enumerate(data):
        if not hasattr(x, '__len__'):
            print(i, "is not iterable")
            missing_data_index.append(i)
            continue
        if x is None:
            print(i, "is None")
            missing_data_index.append(i)
            break
        if len(x) == 0:
            print(i, "row empty")
            missing_data_index.append(i)
            continue
        result.append(x)

    return missing_data_index, np.array(result)
