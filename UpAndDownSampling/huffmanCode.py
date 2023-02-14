import numpy as np

# generate code from tree
def _getcodedict(h, code="", codedict=[]):
	if h["left"] != None:
		codedict = _getcodedict(h["left"], code+"0", codedict)
	if h["right"] != None:
		codedict = _getcodedict(h["right"], code+"1", codedict)

	if h["right"] == None and h["left"] == None:
		currcode = {
			"code": code,			# new code
			"name": h["name"],		# name of original value
			"value": h["value"]		# incidence
		}
		codedict.append(currcode)

	return codedict

# generate code from tree, wrapper
def getcodedict(tree):
	tree.sort(key = lambda x: x.get("value"))
	tree = tree[::-1]
	codedict = _getcodedict(tree[0])
	codedict.sort(key = lambda x: x.get("value"))
	return codedict

# calculate average length of code
def avglength(codedict):
	result = 0
	for elem in codedict:
		result += elem["value"]*len(elem["code"])
	return result

# calculate entropy of signal
def entropy(prob):
	result = 0
	for p in prob:
		result -= p * np.log2(p)
	return result

# pretty print for huffman codes
def printcodedict(codedict, tabwidth = 10):
	head = "code\tname\tvalue".expandtabs(tabwidth)
	print(head + "\n" + "-"*len(head))
	for entry in codedict:
		string = (entry["code"] +"\t"+ entry["name"] +"\t"+ str(entry["value"])).expandtabs(tabwidth)
		print(string)

# debug
def printnodelist(nodelist):
	for node in nodelist:
		print(node["name"])
	print()

# get huffman tree
def huffman_tree(name, prob, debug = False):
	# prepare huffman tree
	huff = []
	for i in range(len(prob)):
		huff.append({
			"name": 	name[i],	# name of variable
			"value": 	prob[i],	# incidence
			"left": 	None,		# left child
			"right": 	None,		# right child
		})

	# start iteration:
	# sort, get two lowest valued items, merge them, repeat with remaining list
	tail = huff
	while(len(tail)>1):
		# sort dict
		tail.sort(key = lambda x: x.get("value"))
		head = tail[:2]
		tail = tail[2:]

		# merge nodes with lowest value
		node = {
			"name": 	f"{head[0]['name']},{head[1]['name']}",
			"value": 	head[0]["value"] + head[1]["value"],
			"left": 	head[0].copy(),
			"right": 	head[1].copy(),
		}
		tail.append(node)
		huff.append(node)

		if debug:
			printnodelist(tail)

	return huff
	
# input
prob = [0.1,0.05,0.25,0.005,0.015,0.03,0.035,0.015,0.15,0.35]
name = ["g1","g2","g3","g4","g5","g6","g7","g8","g9","g10"]

# get tree and code and print some meta data
hufftree = huffman_tree(name, prob)
codedict = getcodedict(hufftree)
print("Huffman code:\n")
printcodedict(codedict)

print("\nAverage code length: \t" + str(avglength(codedict)))
print("Entropy: \t\t\t\t" + str(entropy(prob)) + "\n")

"""
Huffman code:

code      name      value
-------------------------
001010    g4        0.005
00100     g8        0.015
001011    g5        0.015
0000      g6        0.03
0001      g7        0.035
0011      g2        0.05
100       g1        0.1
101       g9        0.15
01        g3        0.25
11        g10       0.35

Average code length: 	2.605
Entropy: 				2.5299651105106378
"""