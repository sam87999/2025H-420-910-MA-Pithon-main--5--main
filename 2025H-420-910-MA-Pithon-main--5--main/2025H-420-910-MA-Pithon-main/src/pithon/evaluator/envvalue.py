"""Définitions des valeurs pour l'évaluateur Pithon."""

from typing import Union,  Callable
from dataclasses import dataclass
from pithon.syntax import ( PiFunctionDef,
)
from pithon.evaluator.envframe import EnvFrame

PrimitiveFunction = Callable[..., 'EnvValue']

@dataclass
class VFunctionClosure:
    """Représente une fermeture de fonction avec son environnement."""
    funcdef: PiFunctionDef
    closure_env: EnvFrame

    def __str__(self) -> str:
        return f"<function {self.funcdef.name} at {id(self)}>"

@dataclass
class VList:
    """Représente une liste de valeurs."""
    value: list['EnvValue']

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return repr(self.value)

@dataclass
class VTuple:
    """Représente un tuple de valeurs."""
    value: tuple['EnvValue', ...]

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return repr(self.value)

@dataclass
class VNumber:
    """Représente un nombre (float)."""
    value: float

    def __str__(self) -> str:
        return str(self.value)


    def __repr__(self) -> str:
        return repr(self.value)

@dataclass
class VBool:
    """Représente une valeur booléenne."""
    value: bool

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return repr(self.value)

@dataclass
class VNone:
    """Représente la valeur None."""
    value: None = None

    def __str__(self) -> str:
        return str(None)

    def __repr__(self) -> str:
        return repr(self.value)

@dataclass
class VString:
    """Représente une chaîne de caractères."""
    value: str

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return repr(self.value)

@dataclass
class VClassDef:
    """Représente une définition de classe avec ses méthodes."""
    name: str
    methods: dict[str, VFunctionClosure]

    def __str__(self) -> str:
        return f"<class {self.name} at {id(self)}>"

@dataclass
class VObject:
    """Représente une instance d'une classe avec ses attributs."""
    class_def: VClassDef
    attributes: dict[str, 'EnvValue']

    def __str__(self) -> str:
        return f"<{self.class_def.name} object at {id(self)}>"

    def __repr__(self) -> str:
        return self.__str__()

@dataclass
class VMethodClosure:
    """Représente une méthode liée à une instance."""
    function: VFunctionClosure
    instance: VObject

    def __str__(self) -> str:
        return f"<method {self.function.funcdef.name} of {self.instance.class_def.name} at {id(self)}>"

    def __repr__(self) -> str:
        return self.__str__()
    
class PiClass:
    def __init__(self, name, body):
        self.name = name
        self.body = body

class PithonAttributeError(Exception):
    """Exception levée lorsqu'un attribut est manquant dans un objet Pithon."""
    pass

class PiInstance:
    def __init__(self, class_def):
        self.class_def = class_def
        self.attributes = {}

    def get_attr(self, name):
        if name in self.attributes:
            return self.attributes[name]
        raise PithonAttributeError(f"Attribut {name} inexistant") 

    def set_attr(self, name, value):
        self.attributes[name] = value

EnvValue = Union[
    VNumber,
    VBool,
    VNone,
    VString,
    VList,
    VTuple,
    VObject,
    VFunctionClosure,
    VMethodClosure,
    VClassDef,
    PrimitiveFunction
]
