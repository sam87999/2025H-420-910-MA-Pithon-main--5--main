class Foo:
    def __init__(self):
        self.x = 42

    def plus_one(self):
        return self.x + 1

foo = Foo()
print(foo.plus_one())