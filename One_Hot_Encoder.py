fruits = ['apple','apple','orange','banana']
drinks = ['soda', 'juice', 'tea', 'coffee']

def one_hot_encode(object):
	set_list = list(set(object))

	#The dimensions of the final one hot encoding are dependent on the length of the initial object and the length of the set of that object 

	object_map = {}

	#We will make a dictionary with keys and values which pairs an integer back to an object from the list

	for i in range(len(set_list)):
		object_map[set_list[i]] = i
      
	one_hot_encode_list = []

	for i in object:
		arr = list(np.zeros(len(set_list), dtype = int)) #This creates a list full of zeros of proper length
		arr[object_map[i]] = 1 #This replaces a 0 with a 1 in places in which we represent an object from the list
		one_hot_encode_list.append(arr)


	print(one_hot_encode_list)

one_hot_encode(fruits)
one_hot_encode(drinks)
