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

import copy, functools, math, fractions, sys, pprint, shutil

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


def young_diagram(partition):
	"""
	Turn a l-partition (list of lists) into a proper 
	young diagram representation. This one only fills the box with col - row.

	Its format is 

	diag [
		part [
			row [
				box (integer),
				box, 
				box,
				...
			],

			row [
				...
			]
		],

		part [
			...
		]
	]

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

def extended_young_diagram(diag, l, k, s):
	"""
	This one extend a normal young diagram with s_content, and annotate boxes with more info 

	box = (sign, content, s_content, c_box, part, row, col)
	"""

	extdiag = []

	has_box = lambda diag, part, row, col: 0 <= part < len(diag) and 0 <= row < len(diag[part]) and 0 <= col < len(diag[part][row])

	for part in range(len(diag)):
		ret_part = []
		for row in range(len(diag[part]) + 1):
			ret_row = []

			# one of the addable box case
			if row == len(diag[part]):
				col = 0

				box = dict()
				box["part"] = part 
				box["row"] = row 
				box["col"] = col 
				box["content"] = col - row 
				box["s_content"] = col - row + s[part]
				box["c_box"] = k * l * box["s_content"] - part

				# addable box
				box["sign"] = "+"

				ret_row = ret_row + [box]

			# other cases, when row < len(diag[part])
			else:
				for col in range(len(diag[part][row]) + 1):
				
					box = dict()
					box["part"] = part 
					box["row"] = row 
					box["col"] = col 
					box["content"] = col - row 
					box["s_content"] = col - row + s[part]
					box["c_box"] = k * l * box["s_content"] - part

					if has_box(diag, part, row, col):
						# removable box
						if col == len(diag[part][row]) - 1 and not has_box(diag, part, row+1, col):
							box["sign"] = "-"
						# regular box
						else:
							box["sign"] = ""

					# addable box
					elif (row == 0 and col == len(diag[part][row])) or (col == len(diag[part][row]) and has_box(diag, part, row-1, col)):
						box["sign"] = "+"

					# no box here
					else:
						continue

					ret_row = ret_row + [box]

			ret_part = ret_part + [ret_row]
		extdiag = extdiag + [ret_part]

	return extdiag

def pretty_young_diagram(young, n, f_box = None):
	"""
	Turn an extended/standard young diagram of original integer n into human readable format

	Args:
		young: a diagram of some l-partition of n 
		n: the original integer being partitioned. we need it to decide the max width of a diagram
		f_box: a function from box position to its content, (box, max_box) -> string

	Returns:
		A human readable string
	"""

	# default printer
	if f_box is None:
		f_box = lambda box, max_box: "[{:>{fill}}]".format(str(box), fill=max_box)

	# find the max number of rows of a diagram
	max_row = functools.reduce(lambda x, y: max(x, y), [len(part) for part in young], 0)

	# find the max width of a box content
	max_box = functools.reduce(lambda x, y: max(x, len(f_box(y, 0))), [box for part in young for row in part for box in row], 0) - 2

	# find the max width of a part
	max_part = (len(str(n-1)) + 4) * n 

	ret = ""

	for row in reversed(range(max_row)):
		for part in range(len(young)):

			# no such row
			if row >= len(young[part]):
				ret += "{:<{fill}}|".format("", fill=max_part+2) 
				continue

			# no such part
			if row == 0 and len(young[part]) == 0:
				ret += "{:<{fill}}|".format("empty", fill=max_part+2) 
				continue

			# normal case
			ret_row = ""
			for box in young[part][row]:
				ret_row += f_box(box, max_box)

			ret += "{:<{fill}}|".format(ret_row, fill=max_part+2) 

		if not row == 0:
			ret += "\n"

	return ret 

def pretty_ext_box(box, max_box):
	if box["sign"] == "":
		return "[{:>{fill}}]".format(str(box["s_content"]), fill=max_box)

	if box["sign"] == "+":
		return " {:>{fill}} ".format(str(box["s_content"]), fill=max_box)

	if box["sign"] == "-":
		return "({:>{fill}})".format(str(box["s_content"]), fill=max_box)

def is_z_box(box, k, z):
	"""
	When k is rational, b is z-box if k * e * s_content_box(b) mod e == k * e * z mod e
	"""

	assert type(k) == fractions.Fraction 

	ke = k.numerator
	e = k.denominator

	return ke * box["s_content"] % e == (ke * z) % e 

def z_signature(extdiag, k, z):
	z_boxes = [box for part in extdiag for row in part for box in row if box["sign"] != "" and is_z_box(box, k, z)]
	z_boxes = sorted(z_boxes, key=lambda box: box["c_box"])

	is_redex = lambda z_boxes, i: (i < len(z_boxes)-1 and z_boxes[i]["sign"] == "-" and z_boxes[i+1]["sign"] == "+") or (0 < i and z_boxes[i-1]["sign"] == "-" and z_boxes[i]["sign"] == "+")
	has_redex = lambda z_boxes: len([box for index, box in enumerate(z_boxes) if is_redex(z_boxes, index)]) > 0
	
	z_boxes = [box for index, box in enumerate(z_boxes) if not is_redex(z_boxes, index)]
	
	while has_redex(z_boxes):
		z_boxes = [box for index, box in enumerate(z_boxes) if not is_redex(z_boxes, index)]
	
	return z_boxes

def normal_young_diagram(extdiag):
	"""
	Transform ext diag back into normal diag
	"""

	diag = []
	for part in range(len(extdiag)):
		part_content = []
		for row in range(len(extdiag[part])):
			row_content = []
			for col in range(len(extdiag[part][row])):
				if extdiag[part][row][col]["sign"] == "+":
					continue
				else: 
					row_content += [extdiag[part][row][col]["content"]]

			if len(row_content) > 0:
				part_content += [row_content]
			else: 
				continue

		diag += [part_content]

	return diag 

def e_transform(extdiag, k, z):
	z_boxes = z_signature(extdiag, k, z)
	diag = normal_young_diagram(extdiag)

	sig = functools.reduce(lambda x, box: x + box["sign"], z_boxes, "")

	n = len([box for part in extdiag for row in part for box in row if box["sign"] == ""])

	print(pretty_young_diagram(extdiag, n+1, pretty_ext_box))
	print("")

	if not "-" in sig:
		# print("{}\n".format(pretty_young_diagram(diag, n+1)))
		return (diag, 0)

	z_box = z_boxes[sig.index("-")]
	del diag[z_box["part"]][z_box["row"]][z_box["col"]]
	if len(diag[z_box["part"]][z_box["row"]]) == 0:
		del diag[z_box["part"]][z_box["row"]]

	# print("{}\n".format(pretty_young_diagram(diag, n+1)))
	return (diag, -1)

def e_transform_all(extdiag, n, k, s, accu, zaccu):

	# if n == 0:
		# return 0

	l = len(extdiag)
	e = k.denominator

	width = shutil.get_terminal_size().columns - 1


	maximum = sys.maxsize * (-1)

	for j in range(e):
		z = j / k.numerator

		z_sig = functools.reduce(lambda x, box: x + box["sign"], z_signature(extdiag, k, z), "")
		title("z   ={} {}\nsig = {}".format(zaccu, int(z), z_sig), width//2)

		(diag, step) = e_transform(extdiag, k, z)
		if step == 0:
			maximum = max(maximum, accu)
			continue

		extdiag_cont = extended_young_diagram(diag, l, k, s)
		maximum = max(maximum, e_transform_all(extdiag_cont, n - 1, k, s, accu+1, "{} {}".format(zaccu, int(z))))

	return maximum

def title(msg, width):
	print("-"*width)
	print(msg)
	print("-"*width)

def test(l, n, k, s):
	
	width = shutil.get_terminal_size().columns - 1

	e = k.denominator
	lams = [extended_young_diagram(young_diagram(partition), l, k, s) for partition in multipartition(l, n)]
	for lam in lams:
		
		# print one lambda
		print("=" * width)
		title("lambda", width)
		print(pretty_young_diagram(lam, n, pretty_ext_box))
		print("")

		max_e = e_transform_all(lam, n, k, s, 0, "") + 1
		title("maximum number of e-tramsform: {}".format(max_e), width)

		# for every z
		# for j in range(e):

			# print("-" * width)
			# z = j / k.numerator
			
			# print signature
			# sig = "".join([box["sign"] for box in z_signature(lam, k, z)])
			# boxes = ",".join(["{}".format(box["s_content"]) for box in z_signature(lam, k, z)])
			# print("{}: {:<{fill}}{}".format(z, sig, boxes, fill=10))
			# print("")

			# # transform to 0
			# (diag, step) = e_transform(x, k, z)
			# extdiag = extended_young_diagram(diag, l, k, s)

			# while step != 0:
				# (diag, step) = e_transform 

			# print(pretty_young_diagram(extdiag, n + step, pretty_ext_box))
		
		input("")
	
# test 2-partition of 3, with k = -1/2, s=(0, -1)
test(2, 3, fractions.Fraction(-1, 2), [0, -1])



