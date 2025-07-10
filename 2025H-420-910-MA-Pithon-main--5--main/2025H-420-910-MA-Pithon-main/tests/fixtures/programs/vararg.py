# Test de Alexandrine Dubé
def f(a, b, *rest):
    res = a + b
    for i in rest:
        res = res + i
    return  res


print(f(1, 2, 3, 4, 5))  # Output: 15
print(f("h","e","llo"," ","w","o","r","ld!"))  # Output: hello world!