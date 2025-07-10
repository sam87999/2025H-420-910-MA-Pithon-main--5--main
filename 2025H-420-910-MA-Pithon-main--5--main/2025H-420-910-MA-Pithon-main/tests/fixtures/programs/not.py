# Test de Dominic Lemay
list=["a","b", "c", "d", "e", "f", "g", "h"]

if not "a" in list:
    print("a n'est pas dans la liste")
else:
    print("a est dans la liste")

if not "a" in list and not "b" in list:
    print("a et b ne sont pas dans la liste")
elif not "a" in list and "b" in list:
    print("a n'est pas dans la liste et b est dans la liste")
elif "a" in list and not "b" in list:
    print("a est dans la liste et b n'est pas dans la liste")
elif "a" in list and "b" in list:
    print("a et b sont dans la liste")

if not "a" in list and not "z" in list:
    print("a et z ne sont pas dans la liste")
elif not "a" in list and "z" in list:
    print("a n'est pas dans la liste et z est dans la liste")
elif "a" in list and not "z" in list:
    print("a est dans la liste et z n'est pas dans la liste")
elif "a" in list and "z" in list:
    print("a et z sont dans la liste")

if not "a" in list and not "b" in list and not "c" in list:
    print("a, b et c ne sont pas dans la liste")
elif not "a" in list and not "b" in list and "c" in list:
    print("a et b ne sont pas dans la liste et c est dans la liste")
elif not "a" in list and "b" in list and not "c" in list:
    print("a n'est pas dans la liste et b est dans la liste et c n'est pas dans la liste")
elif not "a" in list and "b" in list and "c" in list:
    print("a n'est pas dans la liste et b et c sont dans la liste")
elif "a" in list and not "b" in list and not "c" in list:
    print("a est dans la liste et b et c ne sont pas dans la liste")
elif "a" in list and not "b" in list and "c" in list:
    print("a est dans la liste et b n'est pas dans la liste et c est dans la liste")
elif "a" in list and "b" in list and not "c" in list:
    print("a et b sont dans la liste et c n'est pas dans la liste")
elif "a" in list and "b" in list and "c" in list:
    print("a, b et c sont dans la liste")

if not "a" in list and not "b" in list and not "z" in list:
    print("a, b et z ne sont pas dans la liste")
elif not "a" in list and not "b" in list and "z" in list:
    print("a et b ne sont pas dans la liste et z est dans la liste")
elif not "a" in list and "b" in list and not "z" in list:
    print("a n'est pas dans la liste et b est dans la liste et z n'est pas dans la liste")
elif not "a" in list and "b" in list and "z" in list:
    print("a n'est pas dans la liste et b et z sont dans la liste")
elif "a" in list and not "b" in list and not "z" in list:
    print("a est dans la liste et b et z ne sont pas dans la liste")
elif "a" in list and not "b" in list and "z" in list:
    print("a est dans la liste et b n'est pas dans la liste et z est dans la liste")
elif "a" in list and "b" in list and not "z" in list:
    print("a et b sont dans la liste et z n'est pas dans la liste")
elif "a" in list and "b" in list and "z" in list:
    print("a, b et z sont dans la liste")

if not "a" in list and not "y" in list and not "z" in list:
    print("a, y et z ne sont pas dans la liste")
elif not "a" in list and not "y" in list and "z" in list:
    print("a et y ne sont pas dans la liste et z est dans la liste")
elif not "a" in list and "y" in list and not "z" in list:
    print("a n'est pas dans la liste et y est dans la liste et z n'est pas dans la liste")
elif not "a" in list and "y" in list and "z" in list:
    print("a n'est pas dans la liste et y et z sont dans la liste")
elif "a" in list and not "y" in list and not "z" in list:
    print("a est dans la liste et y et z ne sont pas dans la liste")
elif "a" in list and not "y" in list and "z" in list:
    print("a est dans la liste et y n'est pas dans la liste et z est dans la liste")
elif "a" in list and "y" in list and not "z" in list:
    print("a et y sont dans la liste et z n'est pas dans la liste")
elif "a" in list and "y" in list and "z" in list:
    print("a, y et z sont dans la liste")

if not "x" in list and not "y" in list and not "z" in list:
    print("x, y et z ne sont pas dans la liste")
elif not "x" in list and not "y" in list and "z" in list:
    print("x et y ne sont pas dans la liste et z est dans la liste")
elif not "x" in list and "y" in list and not "z" in list:
    print("x n'est pas dans la liste et y est dans la liste et z n'est pas dans la liste")
elif not "x" in list and "y" in list and "z" in list:
    print("x n'est pas dans la liste et y et z sont dans la liste")
elif "x" in list and not "y" in list and not "z" in list:
    print("x est dans la liste et y et z ne sont pas dans la liste")
elif "x" in list and not "y" in list and "z" in list:
    print("x est dans la liste et y n'est pas dans la liste et z est dans la liste")
elif "x" in list and "y" in list and not "z" in list:
    print("x et y sont dans la liste et z n'est pas dans la liste")
elif "x" in list and "y" in list and "z" in list:
    print("x, y et z sont dans la liste")