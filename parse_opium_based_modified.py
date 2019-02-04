import json
count = 0
for i in xrange(1,8):
	file = "output"+str(i)+".txt"
	with open(file, "r") as f:
		twitter = f.readline()
		
		while twitter:
			twitterwords = twitter.split("\t");
			array = ["morphine", "methadone", "buprenorphine", "hydrocodone", "oxycodone" "heroin", "oxycontin", "perc", "percocet","palladone" , "vicodin", "percodan", "tylox" ,"demerol", "oxy", "roxies","opiates", "oxy", "percocet", "percocets", "hydrocodone", "norco",
	    	"norcos", "roxy", "roxies", "roxycodone", "roxicodone", "opana", "opanas", "prozac", "painrelief", "painreliever", "painkillers", "addiction", "opium"]
	    		if any(word in twitterwords[2].lower() for word in array):
	    			a = twitterwords[2].replace('\n',"")
	    			print a
	    			count = count + 1
			twitter = f.readline()

print count



