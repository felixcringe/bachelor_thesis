#!/usr/bin/env python

def isiterable(p_object):
    """Simple check, whether an object is iterable or not."""
    # Source: https://stackoverflow.com/a/4668679
    try:
        it = iter(p_object)
    except TypeError: 
        return False
    return True

#
