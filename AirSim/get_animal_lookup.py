import json

animal_lookup = {0: "deer",
	1: "deer",
	2: "deer",
	3: "fox",
	4: "buffalo",
	5: "bear",
	6: "raccoon",
	7: "rat",
	8: "spider",
	9: "wolf"}

json.dump(animal_lookup, open('animal_lookup.json','w'))