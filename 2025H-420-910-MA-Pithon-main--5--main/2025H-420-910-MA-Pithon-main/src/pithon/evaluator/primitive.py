"""
Fonctions primitives pour l'évaluateur Pithon.
Contient les opérations arithmétiques, comparaisons et fonctions utilitaires de base.
"""

from typing import Any, Type, TypeVar
from pithon.evaluator.envvalue import EnvValue, VList, VNone, VTuple, VNumber, VBool, VString

T = TypeVar('T')
def check_type(obj: Any, mytype: Type[T]) -> T:
    """Vérifie que l'objet est du type attendu, sinon lève une exception."""
    if not isinstance(obj, mytype):
        raise TypeError(f"Type attendu : {mytype.__name__}, obtenu : {type(obj).__name__}")
    return obj

def primitive_add(args: list[EnvValue]):
    """Additionne deux valeurs (nombres, listes, tuples ou chaînes)."""
    a, b = args
    if isinstance(a, VList) and isinstance(b, VList):
        return VList(a.value + b.value)
    if isinstance(a, VTuple) and isinstance(b, VTuple):
        return VTuple(a.value + b.value)
    if isinstance(a, VNumber) and isinstance(b, VNumber):
        return VNumber(a.value + b.value)
    if isinstance(a, VString) and isinstance(b, VString):
        return VString(a.value + b.value)
    raise TypeError(f"Addition non supportée entre {type(a).__name__} et {type(b).__name__}")

def primitive_sub(args: list[EnvValue]):
    """Soustrait deux nombres."""
    a, b = args
    if isinstance(a, VNumber) and isinstance(b, VNumber):
        return VNumber(a.value - b.value)
    raise TypeError(f"Soustraction non supportée entre {type(a).__name__} et {type(b).__name__}")

def primitive_mul(args: list[EnvValue]):
    """Multiplie deux nombres ou répète une séquence (liste, tuple, chaîne)."""
    a, b = args
    # String/List/Tuple repetition: str * int, list * int, tuple * int
    if isinstance(a, VNumber) and isinstance(b, VNumber):
        return VNumber(a.value * b.value)
    # Support for repetition operations
    if isinstance(a, VList) and isinstance(b, VNumber):
        return VList(a.value * int(b.value))
    if isinstance(a, VNumber) and isinstance(b, VList):
        return VList(b.value * int(a.value))
    if isinstance(a, VTuple) and isinstance(b, VNumber):
        return VTuple(a.value * int(b.value))
    if isinstance(a, VNumber) and isinstance(b, VTuple):
        return VTuple(b.value * int(a.value))
    if isinstance(a, VString) and isinstance(b, VNumber):
        return VString(a.value * int(b.value))
    if isinstance(a, VNumber) and isinstance(b, VString):
        return VString(b.value * int(a.value))
    raise TypeError(f"Multiplication non supportée entre {type(a).__name__} et {type(b).__name__}")

def primitive_div(args: list[EnvValue]):
    """Divise deux nombres, lève une erreur si division par zéro."""
    a, b = args
    if isinstance(a, VNumber) and isinstance(b, VNumber):
        if b.value == 0:
            raise ZeroDivisionError("Division par zéro")
        return VNumber(a.value / b.value)
    raise TypeError(f"Division non supportée entre {type(a).__name__} et {type(b).__name__}")

def primitive_mod(args: list[EnvValue]):
    """Calcule le modulo de deux nombres, lève une erreur si division par zéro."""
    a, b = args
    if isinstance(a, VNumber) and isinstance(b, VNumber):
        if b.value == 0:
            raise ZeroDivisionError("Modulo par zéro")
        return VNumber(a.value % b.value)
    raise TypeError(f"Modulo non supporté entre {type(a).__name__} et {type(b).__name__}")

def primitive_eq(args: list[EnvValue]):
    """Teste l'égalité entre deux valeurs."""
    a, b = args
    return VBool(a == b)

def primitive_neq(args: list[EnvValue]):
    """Teste la différence entre deux valeurs."""
    a, b = args
    return VBool(a != b)

def primitive_lt(args: list[EnvValue]):
    """Teste si la première valeur est inférieure à la seconde (nombres ou chaînes)."""
    a, b = args
    if isinstance(a, VNumber) and isinstance(b, VNumber):
        return VBool(a.value < b.value)
    if isinstance(a, VString) and isinstance(b, VString):
        return VBool(a.value < b.value)
    raise TypeError(f"Comparaison '<' non supportée entre {type(a).__name__} et {type(b).__name__}")

def primitive_lte(args: list[EnvValue]):
    """Teste si la première valeur est inférieure ou égale à la seconde (nombres ou chaînes)."""
    a, b = args
    if isinstance(a, VNumber) and isinstance(b, VNumber):
        return VBool(a.value <= b.value)
    if isinstance(a, VString) and isinstance(b, VString):
        return VBool(a.value <= b.value)
    raise TypeError(f"Comparaison '<=' non supportée entre {type(a).__name__} et {type(b).__name__}")

def primitive_gt(args: list[EnvValue]):
    """Teste si la première valeur est supérieure à la seconde (nombres ou chaînes)."""
    a, b = args
    if isinstance(a, VNumber) and isinstance(b, VNumber):
        return VBool(a.value > b.value)
    if isinstance(a, VString) and isinstance(b, VString):
        return VBool(a.value > b.value)
    raise TypeError(f"Comparaison '>' non supportée entre {type(a).__name__} et {type(b).__name__}")

def primitive_gte(args: list[EnvValue]):
    """Teste si la première valeur est supérieure ou égale à la seconde (nombres ou chaînes)."""
    a, b = args
    if isinstance(a, VNumber) and isinstance(b, VNumber):
        return VBool(a.value >= b.value)
    if isinstance(a, VString) and isinstance(b, VString):
        return VBool(a.value >= b.value)
    raise TypeError(f"Comparaison '>=' non supportée entre {type(a).__name__} et {type(b).__name__}")

def primitive_print(args: list[EnvValue]):
    """Affiche la valeur passée en argument."""
    v, = args
    print(v)
    return VNone(value=None)

def primitive_range(args: list[EnvValue]):
    """Crée une liste de nombres dans un intervalle spécifié."""
    if len(args) == 1:
        start = 0
        end = check_type(args[0], VNumber).value
    elif len(args) == 2:
        start = check_type(args[0], VNumber).value
        end = check_type(args[1], VNumber).value
    else:
        raise TypeError("La fonction 'range' attend 1 ou 2 arguments.")
    return VList([VNumber(i) for i in range(int(start), int(end))])

def primitive_str(args: list[EnvValue]):
    """Convertit une valeur en chaîne de caractères."""
    if len(args) != 1:
        raise TypeError("La fonction 'str' attend exactement 1 argument.")
    value = args[0]
    if isinstance(value, VString):
        return value
    if isinstance(value, (VNumber, VBool, VNone, VList, VTuple)):
        return VString(str(value.value))
    else:
        raise TypeError(f"Type non supporté pour 'str': {type(value).__name__}")

def get_primitive_dict():
    """Retourne le dictionnaire des fonctions primitives."""
    return {
        '+': primitive_add,
        '-': primitive_sub,
        '*': primitive_mul,
        '/': primitive_div,
        '%': primitive_mod,
        '==': primitive_eq,
        '!=': primitive_neq,
        '<': primitive_lt,
        '<=': primitive_lte,
        '>': primitive_gt,
        '>=': primitive_gte,
        'print': primitive_print,
        'range': primitive_range,
        'str': primitive_str,
    }