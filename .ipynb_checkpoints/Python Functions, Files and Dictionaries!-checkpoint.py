# num = None
#
# # YOUR CODE HERE
# with open("../assets/travel_plans.txt","r") as f:
#     num = len(f.read())
# print(num)
#
# assert num == 316, "num is not assigned the correct value"



# f = open("../assets/emotion_words.txt","r")
# lines = f.read()
#
# num_words = len(lines.split())
# f.close()

# f = open("../assets/school_prompt.txt","r")
# lines = f.readlines()
# num_lines = len(lines)
# print(num_lines)
# f.close()
#
# f = open("../assets/school_prompt.txt","r")
# beginning_chars = f.read(30)
# print(beginning_chars)
# f.close()

# three = []
#
# # YOUR CODE HERE
# f = open("../assets/school_prompt.txt","r")
# lines = f.readlines()
# print(lines)
# for line in lines:
#     line_str = line.strip().split()
#     three = three + [line_str[2]]
#
# print(three)

p_words = []

# YOUR CODE HERE
f = open("../assets/school_prompt.txt","r")
lines = f.read()
words = lines.split()
for ch in words:
    if ch.find('p') != -1:
        p_words = p_words + [ch]
print(p_words)
f.close()
