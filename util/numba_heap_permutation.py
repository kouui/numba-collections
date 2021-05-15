# origin : https://gist.github.com/kadereub/d3390f75be00df14e65f2a16eb0dc9d9

import numpy as np
import numba as nb

# References
# [1] https://en.wikipedia.org/wiki/Heap%27s_algorithm

@nb.njit
def _factorial(n):
    if n == 1:
        return n
    else:
        return n * _factorial(n-1)


@nb.njit
def numba_heap_permutations(arr, d):
    """
    Generating permutations of an array using Heap's Algorithm
    Args:
        arr (numpy.array): A vector of int/floats which one would like the permutations of
        d (int): The number of permutations, this should in most cases be equal to arr.shape[0]

    Returns:
        (numpy.array): An array of d! rows and d columns, containing all permutations of arr
    """
    d_fact = _factorial(d)
    c = np.zeros(d, dtype=np.int32)
    res = np.zeros(shape=(d_fact, d))
    counter = 0
    i = 0
    res[counter] = arr
    counter += 1
    while i < d:
        if c[i] < i:
            if i % 2 == 0:
                arr[0], arr[i] = arr[i], arr[0]
            else:
                arr[c[i]], arr[i] = arr[i], arr[c[i]]
            # Swap has occurred ending the for-loop. Simulate the increment of the for-loop counter
            c[i] += 1
            res[counter] = arr
            counter += 1
            # Simulate recursive call reaching the base case by bringing setting i to the base case
            i = 0
        else:
            # Calling the func(i+1, A) has ended as the for-loop terminated. 
            # Reset the state and increment i.
            c[i] = 0
            i += 1
    # Return array of d! rows and d columns (all permutations)
    return res
    
    
 # Example use
 numba_heap_perumatations(np.array([1, 2, 3, 4]), 4)
