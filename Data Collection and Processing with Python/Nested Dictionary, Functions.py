import json

# a_string = '\n\n\n{\n "resultCount":25,\n "results": [\n{"wrapperType":"track", "kind":"podcast", "collectionId":10892}]}'
# print(a_string)
# d = json.loads(a_string)
# print("------")
# print(type(d))
# print(d.keys())
# print(d['resultCount'])


# userProfilesString = '{"profiles": \n{"Iman"\n\n: {"tweets": 450, "likes": 2500, "followers": 190, "following": 300},\n\n"Allan"\n\n: {"tweets": 200, "likes": 700, "followers": 150, "following": 100},\n\n"Xinyan"\n\n: {"tweets": 1135, "likes": 3000, "followers": 400, "following": 230},\n\n"Hao"\n\n: {"tweets": 645, "likes": 800, "followers": 300, "following": 500},\n"Harman"\n\n: {"tweets": 300, "likes": 1750, "followers": 200, "following": 400}}}'
# userProfiles = json.loads(userProfilesString)
# print(userProfiles)
# userProfiles['profiles']['Iman']['tweets'] += 5
# userProfiles['profiles']['Iman']['likes'] += 5
# userProfiles['profiles']['Xinyan']['tweets'] += 10
# userProfiles['profiles']['Xinyan']['following'] += 17
# userProfiles['profiles']['Xinyan']['followers'] += 25
# print(userProfiles)
#
# userProfilesString_updated = json.dumps(userProfiles, sort_keys=True, indent=4)
# print(userProfilesString_updated)

inventory = {"Apples": {"Price": 1.50, "Stock": 10},
             "Bananas": {"Price": 1.00, "Stock": 12},
             "Eggs": {"Price": 3.00, "Stock": 7},
             "Milk": {"Price": 3.50, "Stock": 20},
             "Oranges": {"Price": 0.75, "Stock": 35},
             "Avocados": {"Price": 2.50, "Stock": 5}
             }
print(inventory['Avocados']['Price'])

print('Number of Milk cartons in stock is: {}'.format(inventory['Milk']['Stock']))

inventory['Celery'] = {'Price': 1.55, 'Stock': 15}

print(inventory)


def process_shopping_list(inventory,
                          groceryList=["Apples", "Eggs", "Milk", "Avocados", "Broccoli", "Celery", "Cherries"]):
    availableitems = []
    unavailableitems = []
    for grocery in groceryList:
        if grocery in inventory:
            availableitems.append(grocery)
        else:
            unavailableitems.append(grocery)

    total_cost = 0
    for item in availableitems:
        total_cost += inventory[item]['Price']

    return availableitems, total_cost


[availableitems, total_cost] = process_shopping_list(inventory)
print(f'List of available items are {availableitems} and the total cost is: {total_cost}')

