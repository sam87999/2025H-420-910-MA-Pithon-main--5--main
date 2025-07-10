# Test de Maxime Bourret
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def calculate(x, y):
    result = add(x, y)
    result = multiply(result, 2)
    return result

print(calculate(3, 4))
print(add(multiply(2, 3), 5)) 