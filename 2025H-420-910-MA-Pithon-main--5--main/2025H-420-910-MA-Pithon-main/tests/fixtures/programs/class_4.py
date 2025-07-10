y = 0

class Foo:
    def __init__(self):
        self.x = 42

    def get_y(self):
        return y

    def plus_one(self):
        y = self.x + 1
        return y

foo = Foo()
print(y)
print(foo.get_y())
print(foo.plus_one())
print(y)