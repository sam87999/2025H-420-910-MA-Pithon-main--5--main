# Test de Alexandrine Dubé
i = 0
while i < 12:
    if i % 2 == 0:
        i = i + 1
        continue
    if i == 9:
        break
    print(i)
    i = i + 1