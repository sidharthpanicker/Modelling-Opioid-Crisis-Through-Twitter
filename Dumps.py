import json
with open("twitter_drug.txt", "r") as f:
	jsonArray = []
	a = f.readline()
	#print a
	while a:
		dictionary = {}
		dictionary["tweet"] = a
		dictionary["label"] = "None"
		#print dictionary
		jsonArray.append(dictionary)
		a = f.readline()
	length = jsonArray.len() * 0.20
	temp=json.dumps(jsonArray,indent=2)
	print temp


