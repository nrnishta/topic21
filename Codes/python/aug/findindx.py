"""
FINDELMNT Returns the index of an element
List of arguments:
    array: an array, or a list where to search
    element: an element of concern
    err: tolerance
Return:
    touple of indexes

Author:
    Egor Seliunin
    eseliunin@ipfn.tecnico.ulisboa.pt
"""
import numpy as np

def findin(array, element, err):
    """
    Returns the index of an element
    """
    try:
        (x,y) = np.shape(array)
    except ValueError:
        y = 1
        indY = 0
    if np.min(abs(array-element)) < err:
        ind = np.argmin(abs(array-element))
        indY = ind % y
        indX = (ind - indY) / y
        return (indX, indY)
    else:
        raise Exception('No value hase been found')

if __name__=="__main__":
    """
    The scrip is capabel for finding index of required element
    of 1D, or 2D numpy array
    """
    # arrays where to look
    D2 = np.array([[2,1,3],[4,5,6]])
    D1 = np.array([2,1,3])

    element = 1     # element to look for
    err = 0.1       # tolerance

    # examples of function FINDIN
    (indX,indY) = findin(D2, element, err)
    print("2d array:")
    print(indX,indY)

    (indX,indY) = findin(D1, element, err)
    print("1d array:")
    print(indX,indY)
