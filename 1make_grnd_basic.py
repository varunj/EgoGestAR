# up
f = open("train_up.txt", "w+")
x = 320
for y in reversed(range(0,481)):
	f.write("%e" % x + " %e" % y + '\n')

# down
f = open("train_down.txt", "w+")
x = 320
for y in range(0,481):
	f.write("%e" % x + " %e" % y + '\n')

# right
f = open("train_right.txt", "w+")
y = 240
for x in range(0,641):
	f.write("%e" % x + " %e" % y + '\n')

# left
f = open("train_left.txt", "w+")
y = 240
for x in reversed(range(0,641)):
	f.write("%e" % x + " %e" % y + '\n')
