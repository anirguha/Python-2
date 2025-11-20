import json
# a_string = '\n\n\n{\n "resultCount":25,\n "results": [\n{"wrapperType":"track", "kind":"podcast", "collectionId":10892}]}'
# print(a_string)
# d = json.loads(a_string)
# print("------")
# print(type(d))
# print(d.keys())
# print(d['resultCount'])


userProfilesString = '{"profiles": \n{"Iman"\n\n: {"tweets": 450, "likes": 2500, "followers": 190, "following": 300},\n\n"Allan"\n\n: {"tweets": 200, "likes": 700, "followers": 150, "following": 100},\n\n"Xinyan"\n\n: {"tweets": 1135, "likes": 3000, "followers": 400, "following": 230},\n\n"Hao"\n\n: {"tweets": 645, "likes": 800, "followers": 300, "following": 500},\n"Harman"\n\n: {"tweets": 300, "likes": 1750, "followers": 200, "following": 400}}}'
userProfiles = json.loads(userProfilesString)
print(userProfiles)
userProfiles['profiles']['Iman']['tweets'] += 5
userProfiles['profiles']['Iman']['likes'] += 5
userProfiles['profiles']['Xinyan']['tweets'] += 10
userProfiles['profiles']['Xinyan']['following'] += 17
userProfiles['profiles']['Xinyan']['followers'] += 25
print(userProfiles)

userProfilesString_updated = json.dumps(userProfiles, sort_keys=True, indent=4)
print(userProfilesString_updated)
