from ast import expr
from pithon.evaluator.envframe import EnvFrame
from pithon.evaluator.primitive import check_type, get_primitive_dict
from pithon.syntax import (
    PiAssignment, PiBinaryOperation, PiNumber, PiBool, PiStatement, PiProgram, PiSubscript, PiVariable,
    PiIfThenElse, PiNot, PiAnd, PiOr, PiWhile, PiNone, PiList, PiTuple, PiString,
    PiFunctionDef, PiFunctionCall, PiFor, PiBreak, PiContinue, PiIn, PiReturn, PiClassDef, 
    PiInstanceCreation, PiAttribute, PiAttributeAccess, PiAttributeAssignment
)
from pithon.evaluator.envvalue import EnvValue, PiClass, PiInstance, VFunctionClosure, VList, VNone, VTuple, VNumber, VBool, VString

class PithonRuntimeError(Exception): pass
class PithonTypeError(PithonRuntimeError): pass
class PithonNameError(PithonRuntimeError): pass
class PithonAttributeError(PithonRuntimeError): pass

def initial_env() -> EnvFrame:
    env = EnvFrame()
    env.vars.update(get_primitive_dict())
    return env

def lookup(env: EnvFrame, name: str) -> EnvValue:
    return env.lookup(name)

def insert(env: EnvFrame, name: str, value: EnvValue) -> None:
    env.insert(name, value)

def evaluate(node: PiProgram, env: EnvFrame) -> EnvValue:
    if isinstance(node, list):
        last_value = VNone(value=None)
        for stmt in node:
            last_value = evaluate_stmt(stmt, env)
        return last_value
    elif isinstance(node, PiStatement):
        return evaluate_stmt(node, env)
    else:
        raise TypeError(f"Type de nœud non supporté : {type(node)}")

def evaluate_stmt(node: PiStatement, env: EnvFrame) -> EnvValue:
    if isinstance(node, PiNumber):
        return VNumber(node.value)
    elif isinstance(node, PiBool):
        return VBool(node.value)
    elif isinstance(node, PiNone):
        return VNone(node.value)
    elif isinstance(node, PiString):
        return VString(node.value)
    elif isinstance(node, PiList):
        elements = [evaluate_stmt(e, env) for e in node.elements]
        return VList(elements)
    elif isinstance(node, PiTuple):
        elements = tuple(evaluate_stmt(e, env) for e in node.elements)
        return VTuple(elements)
    elif isinstance(node, PiVariable):
        return lookup(env, node.name)
    elif isinstance(node, PiBinaryOperation):
        fct_call = PiFunctionCall(
            function=PiVariable(name=node.operator),
            args=[node.left, node.right]
        )
        return evaluate_stmt(fct_call, env)
    elif isinstance(node, PiAssignment):
        value = evaluate_stmt(node.value, env)
        insert(env, node.name, value)
        return value
    elif isinstance(node, PiIfThenElse):
        cond = evaluate_stmt(node.condition, env)
        cond = check_type(cond, VBool)
        branch = node.then_branch if cond.value else node.else_branch
        last_value = evaluate(branch, env)
        return last_value
    elif isinstance(node, PiNot):
        operand = evaluate_stmt(node.operand, env)
        _check_valid_piandor_type(operand)
        return VBool(not operand.value)
    elif isinstance(node, PiAnd):
        left = evaluate_stmt(node.left, env)
        _check_valid_piandor_type(left)
        if not left.value:
            return left
        right = evaluate_stmt(node.right, env)
        _check_valid_piandor_type(right)
        return right
    elif isinstance(node, PiOr):
        left = evaluate_stmt(node.left, env)
        _check_valid_piandor_type(left)
        if left.value:
            return left
        right = evaluate_stmt(node.right, env)
        _check_valid_piandor_type(right)
        return right
    elif isinstance(node, PiWhile):
        return _evaluate_while(node, env)
    elif isinstance(node, PiFunctionDef):
        closure = VFunctionClosure(node, env)
        insert(env, node.name, closure)
        return VNone(value=None)
    elif isinstance(node, PiReturn):
        value = evaluate_stmt(node.value, env)
        raise ReturnException(value)
    elif isinstance(node, PiFunctionCall):
        return _evaluate_function_call(node, env)
    elif isinstance(node, PiFor):
        return _evaluate_for(node, env)
    elif isinstance(node, PiBreak):
        raise BreakException()
    elif isinstance(node, PiContinue):
        raise ContinueException()
    elif isinstance(node, PiIn):
        return _evaluate_in(node, env)
    elif isinstance(node, PiSubscript):
        return _evaluate_subscript(node, env)
    elif isinstance(node, PiClassDef):
        insert(env, node.name, PiClass(node.name, node.body))
        return VNone()
    elif isinstance(node, PiInstanceCreation):
        class_def = lookup(env, node.class_name)
        if not isinstance(class_def, PiClass):
            raise PithonTypeError(f"{node.class_name} n'est pas une classe")
        return PiInstance(class_def)
    elif isinstance(node, PiAttributeAccess):
        obj = evaluate_stmt(node.object_expr, env)
        if not isinstance(obj, PiInstance):
            raise PithonTypeError("Accès à un attribut sur un objet non instancié")
        return obj.get_attr(node.attribute)
    elif isinstance(node, PiAttributeAssignment):
        obj = evaluate_stmt(node.object_expr, env)
        if not isinstance(obj, PiInstance):
            raise PithonTypeError("Affectation d'un attribut sur un objet non instancié")
        val = evaluate_stmt(node.value_expr, env)
        obj.set_attr(node.attribute, val)
        return val
    else:
        raise TypeError(f"Type de nœud non supporté : {type(node)}")

def _check_valid_piandor_type(obj):
    if not isinstance(obj, (VBool, VNumber, VString, VNone, VList, VTuple)):
        raise TypeError(f"Type non supporté pour l'opérateur 'and': {type(obj).__name__}")

def _evaluate_while(node: PiWhile, env: EnvFrame) -> EnvValue:
    last_value = VNone(value=None)
    while True:
        cond = evaluate_stmt(node.condition, env)
        cond = check_type(cond, VBool)
        if not cond.value:
            break
        try:
            last_value = evaluate(node.body, env)
        except BreakException:
            break
        except ContinueException:
            continue
    return last_value

def _evaluate_for(node: PiFor, env: EnvFrame) -> EnvValue:
    iterable_val = evaluate_stmt(node.iterable, env)
    if not isinstance(iterable_val, (VList, VTuple)):
        raise TypeError("La boucle for attend une liste ou un tuple.")
    last_value = VNone(value=None)
    iterable = iterable_val.value
    for item in iterable:
        env.insert(node.var, item)
        try:
            last_value = evaluate(node.body, env)
        except BreakException:
            break
        except ContinueException:
            continue
    return last_value

def _evaluate_subscript(node: PiSubscript, env: EnvFrame) -> EnvValue:
    collection = evaluate_stmt(node.collection, env)
    index = evaluate_stmt(node.index, env)
    if isinstance(collection, VList):
        idx = check_type(index, VNumber)
        return collection.value[int(idx.value)]
    elif isinstance(collection, VTuple):
        idx = check_type(index, VNumber)
        return collection.value[int(idx.value)]
    elif isinstance(collection, VString):
        idx = check_type(index, VNumber)
        return VString(collection.value[int(idx.value)])
    else:
        raise TypeError("L'indexation n'est supportée que pour les listes, tuples et chaînes.")

def _evaluate_in(node: PiIn, env: EnvFrame) -> EnvValue:
    container = evaluate_stmt(node.container, env)
    element = evaluate_stmt(node.element, env)
    if isinstance(container, (VList, VTuple)):
        return VBool(element in container.value)
    elif isinstance(container, VString):
        if isinstance(element, VString):
            return VBool(element.value in container.value)
        else:
            return VBool(False)
    else:
        raise TypeError("'in' n'est supporté que pour les listes et chaînes.")

class ReturnException(Exception):
    def __init__(self, value):
        self.value = value

class BreakException(Exception):
    pass

class ContinueException(Exception):
    pass

def _evaluate_function_call(node: PiFunctionCall, env: EnvFrame) -> EnvValue:
    func_val = evaluate_stmt(node.function, env)
    args = [evaluate_stmt(arg, env) for arg in node.args]
    if callable(func_val):
        return func_val(args)
    if not isinstance(func_val, VFunctionClosure):
        raise TypeError("Tentative d'appel d'un objet non-fonction.")
    funcdef = func_val.funcdef
    closure_env = func_val.closure_env
    call_env = EnvFrame(parent=closure_env)
    for i, arg_name in enumerate(funcdef.arg_names):
        if i < len(args):
            call_env.insert(arg_name, args[i])
        else:
            raise TypeError("Argument manquant pour la fonction.")
    if funcdef.vararg:
        varargs = VList(args[len(funcdef.arg_names):])
        call_env.insert(funcdef.vararg, varargs)
    elif len(args) > len(funcdef.arg_names):
        raise TypeError("Trop d'arguments pour la fonction.")
    result = VNone(value=None)
    try:
        for stmt in funcdef.body:
            result = evaluate_stmt(stmt, call_env)
    except ReturnException as ret:
        return ret.value
    return result
