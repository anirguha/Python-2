# stopwords = ['to', 'a', 'for', 'by', 'an', 'am', 'the', 'so', 'it', 'and', 'The']
# sent = "The water earth and air are vital"
# acro = ""
# sent_list = sent.split()
# for word in sent_list:
#     if word not in stopwords:
#         acro = acro + word[:2].upper() + "."
# acro = acro[:-1]
# print(acro)
#
# p_phrase = "Madam"
# r_phrase = ""
# for idx in range(len(p_phrase)):
#     r_phrase = p_phrase[idx] + r_phrase
# if p_phrase.lower() == r_phrase.lower():
#     print('Palindrome')
# else:
#     print('Not a Palindrome')

inventory = ["shoes, 12, 29.99", "shirts, 20, 9.99", "sweatpants, 25, 15.00", "scarves, 13, 7.75"]

for store in inventory:
    item_quantity_price = store.split(",")
    item = item_quantity_price[0].strip()
    quantity = item_quantity_price[1].strip()
    price = item_quantity_price[2].strip()
    output = "The store has {} {}, each for {} USD.".format(quantity, item, price)
    print(output)
