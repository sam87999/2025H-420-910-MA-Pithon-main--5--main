def foo(a, b):
    def bar():
        return a + b
    return bar

f1 = foo(1, 2)
print(f1())

f2 = foo(3, 4)
print(f2())