class EnvFrame:
    """
    Représente un cadre d'environnement pour stocker des variables avec un lien vers un parent.
    """
    def __init__(self, parent=None):
        """
        Initialise un nouvel environnement, éventuellement avec un parent.
        """
        self.vars = {}
        self.parent: EnvFrame | None = parent

    def lookup(self, name):
        """
        Recherche la valeur d'une variable par son nom dans l'environnement courant ou ses parents.
        """
        if name in self.vars:
            return self.vars[name]
        elif self.parent is not None:
            return self.parent.lookup(name)
        else:
            raise NameError(f"Variable '{name}' non définie.")

    def insert(self, name, value):
        """
        Insère ou met à jour une variable dans l'environnement courant.
        """
        self.vars[name] = value

    def copy_shallow(self):
        """
        Retourne une copie superficielle de l'environnement (variables copiées, même parent).
        """
        newf = EnvFrame(self.parent)
        newf.vars = self.vars.copy()
        return newf