# try:
#     for i in range(5):
#         print(1.0 / (3-i))
# except Exception as error_inst:
#     print("Got an error", error_inst)


students = [('Timmy', 95, 'Will pass'), ('Martha', 70), ('Betty', 82, 'Will pass'), ('Stewart', 50, 'Will not pass'), ('Ashley', 68), ('Natalie', 99, 'Will pass'), ('Archie', 71), ('Carl', 45, 'Will not pass')]

passing = {'Will pass': 0, 'Will not pass': 0}

for tup in students:
    try:
        if tup[2] == 'Will pass':
             passing['Will pass'] += 1
        elif tup[2] == 'Will not pass':
                passing['Will not pass'] += 1
    except IndexError:
        continue

print(passing)

nums = [5, 9, '4', 3, 2, 1, 6, 5, '7', 4, 3, 2, 6, 7, 8, '0', 3, 4, 0, 6, 5, '3', 5, 6, 7, 8, '3', '1', 5, 6, 7, 9, 3,
        2, 5, 6, '9', 2, 3, 4, 5, 1]

plus_four = []

for num in nums:
    try:
        plus_four.append(num + 4)
    except TypeError:
        plus_four.append('Error')

print(plus_four)
