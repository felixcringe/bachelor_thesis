#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""(i)mutable Namespace object.
derived from argparse.Namespace"""

import collections
import numpy as np

class _AttributeHolder(object):
	"""Abstract base class that provides __repr__.

	The __repr__ method returns a string in the format::
		ClassName(attr=name, attr=name, ...)
	The attributes are determined either by a class-level attribute,
	'_kwarg_names', or by inspecting the instance __dict__.
	"""

	def __repr__(self):
		type_name = type(self).__name__
		arg_strings = []
		for arg in self._get_args():
			arg_strings.append(repr(arg))
		for name, value in self.iteritems():
			arg_strings.append('%s=%r' % (name, value))
		return '%s(\n    %s\n)' % (type_name, ',\n    '.join(arg_strings))

	def _get_kwargs(self):
		return sorted(self.items())

	def _get_args(self):
		return []

	def keys(self):
		return list(self.iterkeys())

	def iterkeys(self):
		return (key for key in self.__dict__.keys() if key != '__updateable__')

	def values(self):
		return list(self.itervalues())

	def itervalues(self):
		return (key for key in self.__dict__.values() if key != '__updateable__')

	def items(self):
		return list(self.iteritems())

	def iteritems(self):
		return ((key, value) for key, value in self.__dict__.iteritems() if key != '__updateable__')



class Namespace(_AttributeHolder):
	"""Simple object for storing attributes.

	Implements equality by attribute names and values, and provides a simple
	string representation.

	special keywords:
	__data_dict__: You may pass a dict, if some keys are not compatible to
	    keyword arguments (i.e. non-string key).
	__updateable__:
		default: False
		If True, the attributes of this Namespace are changeable. This does
		_not_ effect the attributes of the attribute. e.g. if the attribute
		is an numpy array, you can still change the values by ns.x[:] = 42
	"""

	def __init__(self, __data_dict__ = None, __updateable__ = False, **kwargs):
		self.__updateable__ = __updateable__

		if __data_dict__ is None:
			data_dict = {}
			# Tip: Don't use {} as default value instead of None.
		else:
			data_dict = __data_dict__

		# join data_dict and keyword arguments
		data_dict.update(kwargs)

		for key in data_dict.keys():
			if key in dir(Namespace):
				raise ValueError("'%s' in 'Namespace' object is a reserved key" % key)

			if isinstance(data_dict[key], dict):
				self.__dict__[key] = Namespace(data_dict[key], __updateable__=__updateable__)
			elif isinstance(data_dict[key], list):
				self.__dict__[key] = tuple(data_dict[key])
			else:
				self.__dict__[key] = data_dict[key]


	# just to set the state of __updateable__
	__updateable__ = True

	__hash__ = None

	def __eq__(self, other):
		return vars(self) == vars(other)

	def __ne__(self, other):
		return not (self == other)

	def __contains__(self, key):
		return key in self.__dict__

	def __getitem__(self, key):
		return getattr(self, key)

	def __setitem__(self, key, value):
		setattr(self, key, value)

	def __setattr__(self, key, value):
		if not self.__updateable__ and key in self:
			if isinstance(value, np.ndarray) and isinstance(self.__dict__[key], np.ndarray):
				# np.arrays are allowed to be updatet if they have the same type and shape
				if not value.shape == self.__dict__[key].shape:
					raise ValueError("Can not update '%s' because shapes do not match and this Namespace is not __updateable__" % key, Namespace)
				elif not self.__dict__[key].dtype is self.__dict__[key].dtype:
					raise ValueError("Can not update '%s' because dtypes do not match and this Namespace is not __updateable__" % key, Namespace)
				else:
					self.__dict__[key] = value
			else:
				raise ValueError("Can not update '%s' because this Namespace is not __updateable__" % key, Namespace)
		else:
			self.__dict__[key] = value

	def get(self, key, default = None):
		"""N.get(key[,default]) -> N[key] if key in N, else default.  default defaults to None."""
		if key in self:
			return self[key]
		else:
			return default
