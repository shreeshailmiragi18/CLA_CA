import numpy as np

def create_split(attr_values_disc: np.ndarray, bim: int) -> int:
    """
    Algorithm 3: Createsplit (Valley Detection)
    Returns split point index (1 to bim).
    """
    num = np.zeros(bim)
    for val in attr_values_disc:
        idx = int(val) - 1
        if 0 <= idx < bim:
            num[idx] += 1

    s_list = []
    # Loop from region 2 to bim-1 (index 1 to bim-2)
    for i in range(1, bim - 1):
        p = num[i + 1] - num[i]
        q = num[i - 1] - num[i]

        if q > 0 and p > 0:
            val = p + q
            # Add with 1-based index (i+1)
            s_list.append({'val': val, 'index': i + 1})

    if not s_list:
        return -1

    # Return index maximizing p+q
    best_point = max(s_list, key=lambda x: x['val'])
    return best_point['index']
