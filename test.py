a1 = [1, 2, 3, 4, 5, 6]
a2 = [7, 8, 9]

pairs = zip(*[iter(a1)]*2)
for (i, j), k in zip(pairs, a2):
    print(i, j, k)
