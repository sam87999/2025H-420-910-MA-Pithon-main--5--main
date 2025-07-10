# Test de Maxime Bourret
def check_number(x):
    if x > 0:
        if x % 2 == 0:
            return "positive even"
        else:
            return "positive odd"
    else:
        if x == 0:
            return "zero"
        else:
            return "negative"

print(check_number(4))
print(check_number(7))
print(check_number(0)) 