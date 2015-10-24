""" combinatorics.py

SUMMARY
=======

Provides some utility functions for Huijun Zhao to compute stuff that I don't understand.

1. partition(n)
2. multipartition(l, n)
3. balls_in_bins(balls, bins)
4. young_diagram(partition)
5. pretty_young_diagram(diag, n)

AUTHOR
======

Hanwen Wu <steinwaywhw@gmail.com>

"""

import copy, functools, math

def partition(n):
	"""
	Partition an positive integer n into non-incresing addends
	
	Args: 
		n: the positive integer

	Returns:
		a list of partitions (which again is a list of non-incresing addends)
	"""
	assert type(n) is int 
	assert n > 0

	return _partition(n, n)


def _partition(n, top):
	"""
	Partition an integer n into addends smaller than or equal to top

	Args: 
		n: the number being partitioned 
		top: the largest possible addends of a partition of n

	Returns: 
		a list of partitions (which again is a list of non-incresing addends)
	"""

	if top == 1:
		return [[1] * n]

	if n == top:
		return [[top]] + _partition(n, top - 1)
	else:
		return [[top] + x for x in _partition(n - top, min(n - top, top))] + _partition(n, top - 1)

def multipartition(l, n):
	"""
	Solve l-partition n

	Args:
		l: the l
		n: the number being partitioned 

	Returns:
		a list of l-partitions
	"""

	assert type(l) is int and type(n) is int 
	assert l > 0 and n > 0

	ret = []
	for p in partition(n):
		ret = ret + balls_in_bins(p, [[]] * l)

	return ret 

def balls_in_bins(balls, bins):
	ret = []
	_balls_in_bins(balls, bins, ret)

	# compute a hash for a partition by computing the weighted length of each part in a partition
	# [[1,1,1],[1,1],[1]]: compute each length to get
	# => [3, 2, 1]: and reduce it into 
	# => 321
	# hashing = lambda a, b: a * 10 + b 

	# give a order to all possible partitions by comparing hash
	# sorting_key = lambda partition: functools.reduce(hashing, [len(part) for part in partition], 0)

	# sort it
	# ret = sorted(ret, key=sorting_key)

	# remove duplicate
	return [p for index, p in enumerate(ret) if p not in ret[:index]]


def _balls_in_bins(balls, bins, ret):
	"""
	Throw a list of balls into n bins 

	Args:
		balls: a list of balls 
		bins: initial bins, which should be a list of lists [[], [], [], ...] 
		ret: accumulation of the current result

	Returns:
		a list of resuts, with each one being a list of bins (which is again a list of balls)
	"""

	assert type(balls) is list
	assert type(bins) is list and len(bins) > 0
	assert type(ret) is list 

	if len(balls) == 0:
		ret += [bins]
		return bins

	for b in range(len(bins)):
		_balls_in_bins(balls[1:], _ball_in_bin(balls[0], bins, b), ret)


def _ball_in_bin(ball, bins, index):
	"""
	Put a ball in bins[index], return a copy

	Args:
		ball: the ball 
		bins: a list of lists
		index: the bin to put ball in 

	Returns: 
		a new copy of the new bins (deep copy)
	"""

	assert type(bins) is list 
	assert type(index) is int and 0 <= index < len(bins)

	retbins = copy.deepcopy(bins)
	retbins[index] = retbins[index] + [ball]

	return retbins

# y = balls_in_bins([1,1,1], [[]] * 2)

def young_diagram(partition):
	"""
	Turn a l-partition (list of lists) into a proper 
	young diagram representation

	e.g. [[2, 1], [3]] => [[ [ [],[] ],  [   [[],[],[]] ]]
							 [ []    ]], 

	e.g. [[], [1]] => [ [], [[ []  ]]]

	Its format is diag(part(row(boxes())))

	Args: 
		partition: one l-partition, e.g. [[2, 1], [3]]. Each part in the partition should be in decreasing order

	Returns:
		a young diagram representation
	"""

	# assert
	for part in partition:
		assert part == sorted(part, reverse=True)

	# turn a positive integer at row into boxes
	to_boxes = lambda n, row: [col - row for col in range(n)]

	# turn a part of the partition into rows
	to_rows = lambda part: [to_boxes(n, row) for row, n in enumerate(part)]

	return [to_rows(part) for part in partition]


def pretty_young_diagram(diag, n):
	"""
	Turn a young diagram of original integer n into human readable format

	Args:
		diag: a diagram of some l-partition of n 
		n: the original integer being partitioned. we need it to decide the max width of a diagram

	Returns:
		A human readable string
	"""

	# find the max number of rows of a diagram
	max_row = functools.reduce(lambda x, y: max(x, y), [len(part) for part in diag], 0)
	max_box = functools.reduce(lambda x, y: max(x, len(str(y))), [box for part in diag for row in part for box in row], 0)

	# find the max width of a part
	max_part = (len(str(n-1)) + 3) * n

	pretty_box = lambda box: "[{:>{fill}}]".format(box, fill=max_box)
	pretty_row = lambda row: functools.reduce(lambda x, y: x + pretty_box(y), row, "")

	ret = "=" * (max_part * len(diag) + 2) + "\n"

	for row in range(max_row):
		for part in range(len(diag)):
			if row < len(diag[part]):
				ret += "{:<{fill}}".format(pretty_row(diag[part][row]), fill=max_part+2) 
			elif row == 0 and len(diag[part]) == 0 :
				ret += "{:<{fill}}".format("empty", fill=max_part+2) 
			else:
				ret += "{:<{fill}}".format("", fill=max_part+2) 

		ret += "\n"

	return ret 



def test(l, n):
	y = [young_diagram(partition) for partition in multipartition(l, n)]
	for x in y:
		print(pretty_young_diagram(x, n))
		input("")
	
test(2, 9)



