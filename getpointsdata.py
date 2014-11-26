import json
import re
import os, os.path
import csv

## get the relevant quote from a json file
def getquote(text):
	quotes = re.findall(ur'[^"^\u201c]*["\u201d]', text)
	for quote in quotes:
		return quote

# example quote extraction
json_data = open('tosdr.org/points/example.json')
data = json.load(json_data)
topic = data['topics']
print topic[0]
tldr = data['tosdr']['tldr']
getquote(tldr)

# create list of all points in the folder to be analysed
pointlist = []
path='tosdr.org/points'
dirList=os.listdir(path)
for fname in dirList:
	if fname.endswith(".json"):
		pointlist.append(fname)

# find the topic, quote and rating for each point and write to a csv file
for point in pointlist:
	json_data = open('tosdr.org/points/%s' % point)
	data = json.load(json_data)
	if 'topics' in data:
		if 'tosdr' in data:
			tosdr = data['tosdr']
			if 'tldr' in tosdr:
				topic = data['topics'][0]
				topic = topic.encode('UTF-8', 'replace')
				tldr = tosdr['tldr']
				tldr = tldr.encode('UTF-8', 'replace')
				tldr = str(tldr)
				tldr = tldr.replace(",", "")
#				tldr = re.findall(ur'[^"^\u201c]*["\u201d]', tldr) - in case you want to limit to direct quotations from the policy
				print tldr
				if 'score' in tosdr:
					rating = str(tosdr['point'])
					print rating
					result = "%s,%s,%s\n" % (topic, tldr, rating)
#					result = result.encode('UTF-8', 'replace')
					with open("tostraining5.csv", "a") as archive:
						archive.write(result)
					print "archived entry on %s" % topic
