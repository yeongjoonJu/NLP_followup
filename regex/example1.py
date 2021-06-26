import re

# [abc] : a or b or c (a|b|c)
# [2-5] : 2~5
# [^2-5c-e] : expect to 2~5, c~e
#  () : group
# x? : 1 or less
# x+ : 1 more
# x* : nothing or 1 more
# x{n,m} : repeat n~m times
# . : any characters
# \s : white space
# \w : alphanumeric + '_'
# \d : [0-9]

example_sent = """
Hello Ki, I would like to introduce regular expression in this section.
~~
Thank you!
Sincerely
Ki: +82-10-1234-5678
Content jiu 02)1234-5678
"""

regex = r"([\w]+\s*:?)?[\s]*(\(?\+?[0-9]{,3}\)?)?\-?[0-9]{2,3}\)?\-?[0-9]{3,4}\-?[0-9]{4}"

print(re.sub(regex, "REMOVED", example_sent))

x = """abcdef
12345
ab12
a1bc2d
12ab
a1b
la2
a1
1a
hfdfasf"""

regex = r"([a-z])[0-9]+([a-z])"
to = r'\1\2'

y = '\n'.join([re.sub(regex, to, x_i) for x_i in x.split('\n')])
print(y)