p_words = []

# YOUR CODE HERE
f = open("assets/school_prompt.txt","r")
lines = f.read()
words = lines.split()
for ch in words:
    if ch.find('p') != -1:
        p_words = p_words + [ch]
print(p_words)
f.close()