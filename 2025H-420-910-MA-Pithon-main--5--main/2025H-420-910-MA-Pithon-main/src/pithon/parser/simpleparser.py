import ast

from pithon.syntax import (
    PiAssignment, PiBinaryOperation, PiNumber, PiBool, PiVariable, PiIfThenElse,
    PiNot, PiAnd, PiOr, PiWhile, PiExpression, PiNone, PiList, PiTuple,
    PiString, PiFunctionDef, PiFunctionCall, PiFor, PiBreak, PiContinue, PiIn,
    PiReturn, PiSubscript, PiClassDef, PiAttribute, PiAttributeAssignment
)

class SimpleParser(ast.NodeVisitor):
    """
    Un parseur simple pour le langage Pithon.
    Il utilise le module ast de Python pour analyser le code source et
    retourner un arbre syntaxique abstrait (AST) simplifié propre à Pithon.
    """
    def parse(self, source_code: str):
        tree = ast.parse(source_code)
        return [self.visit(stmt) for stmt in tree.body]

    def visit_Expr(self, node: ast.Expr) -> PiExpression:
        return self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> PiAssignment | PiAttributeAssignment:
        if len(node.targets) != 1:
            raise ValueError("Seule l'affectation simple est prise en charge.")
        target = node.targets[0]
        value = self.visit(node.value)
        
        if isinstance(target, ast.Name):
            # Simple variable assignment
            return PiAssignment(name=target.id, value=value)
        elif isinstance(target, ast.Attribute):
            # Attribute assignment
            obj = self.visit(target.value)
            return PiAttributeAssignment(object=obj, attr=target.attr, value=value)
        else:
            raise ValueError("Les affectations ne peuvent être faites qu'à des variables ou des attributs.")

    def visit_BinOp(self, node: ast.BinOp) -> PiBinaryOperation:
        left = self.visit(node.left)
        right = self.visit(node.right)
        operator = self.operator_symbol(node.op)
        return PiBinaryOperation(left=left, operator=operator, right=right)

    def visit_Name(self, node: ast.Name) -> PiVariable:
        return PiVariable(name=node.id)

    def visit_Constant(self, node: ast.Constant) -> PiExpression:
        if node.value is None:
            return PiNone(value=None)
        if isinstance(node.value, bool):
            return PiBool(value=node.value)
        elif isinstance(node.value, int) or isinstance(node.value, float):
            return PiNumber(value=node.value)
        elif isinstance(node.value, str):
            return PiString(value=node.value)
        else:
            raise ValueError("Seuls les littéraux numériques, booléens, chaînes ou None sont autorisés.")

    def visit_List(self, node: ast.List) -> PiList:
        elements = [self.visit(elt) for elt in node.elts]
        return PiList(elements=elements)

    def visit_Tuple(self, node: ast.Tuple) -> PiTuple:
        elements = tuple(self.visit(elt) for elt in node.elts)
        return PiTuple(elements=elements)

    def visit_If(self, node: ast.If) -> PiIfThenElse:
        condition = self.visit(node.test)
        then_branch = [self.visit(stmt) for stmt in node.body]
        else_branch = [self.visit(stmt) for stmt in node.orelse] if node.orelse else []
        return PiIfThenElse(condition=condition, then_branch=then_branch, else_branch=else_branch)

    def visit_IfExp(self, node: ast.IfExp) -> PiIfThenElse:
        # Pour les expressions ternaires (x if cond else y)
        condition = self.visit(node.test)
        then_branch = [self.visit(node.body)]
        else_branch = [self.visit(node.orelse)]
        return PiIfThenElse(condition=condition, then_branch=then_branch, else_branch=else_branch)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> PiExpression:
        if isinstance(node.op, ast.Not):
            operand = self.visit(node.operand)
            return PiNot(operand=operand)
        else:
            raise ValueError("Seul l'opérateur 'not' unaire est supporté.")

    def visit_BoolOp(self, node: ast.BoolOp) -> PiExpression:
        # Support left-associative nesting for multiple operands
        if isinstance(node.op, ast.And):
            expr = self.visit(node.values[0])
            for value in node.values[1:]:
                expr = PiAnd(left=expr, right=self.visit(value))
            return expr
        elif isinstance(node.op, ast.Or):
            expr = self.visit(node.values[0])
            for value in node.values[1:]:
                expr = PiOr(left=expr, right=self.visit(value))
            return expr
        else:
            raise ValueError("Seuls les opérateurs 'and' et 'or' sont supportés.")

    def visit_While(self, node: ast.While) -> PiWhile:
        condition = self.visit(node.test)
        body = [self.visit(stmt) for stmt in node.body]
        return PiWhile(condition=condition, body=body)

    def visit_For(self, node: ast.For) -> PiFor:
        if not isinstance(node.target, ast.Name):
            raise ValueError("La variable de boucle doit être un nom simple.")
        var = node.target.id
        iterable = self.visit(node.iter)
        body = [self.visit(stmt) for stmt in node.body]
        return PiFor(var=var, iterable=iterable, body=body)

    def visit_Break(self, node: ast.Break) -> PiBreak:
        return PiBreak()

    def visit_Continue(self, node: ast.Continue) -> PiContinue:
        return PiContinue()

    def visit_Compare(self, node: ast.Compare) -> PiExpression:
        if len(node.ops) == 1 and isinstance(node.ops[0], ast.In):
            element = self.visit(node.left)
            container = self.visit(node.comparators[0])
            return PiIn(element=element, container=container)
        elif len(node.ops) == 1:
            left = self.visit(node.left)
            right = self.visit(node.comparators[0])
            operator = self.operator_symbol(node.ops[0])
            return PiBinaryOperation(left=left, operator=operator, right=right)
        else:
            raise ValueError("Seules les comparaisons simples sont supportées.")

    def visit_Call(self, node: ast.Call) -> PiExpression:
        func = self.visit(node.func)
        args = [self.visit(arg) for arg in node.args]
        return PiFunctionCall(function=func, args=args)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> PiFunctionDef:
        name = node.name
        arg_names = []
        vararg = None
        for arg in node.args.args:
            arg_names.append(arg.arg)
        if node.args.vararg:
            vararg = node.args.vararg.arg
        body = [self.visit(stmt) for stmt in node.body]
        return PiFunctionDef(name=name, arg_names=arg_names, vararg=vararg, body=body)

    def visit_Return(self, node: ast.Return) -> PiReturn:
        value = self.visit(node.value) if node.value else PiNone(value=None)
        return PiReturn(value=value)

    def visit_Subscript(self, node: ast.Subscript) -> PiSubscript:
        collection = self.visit(node.value)
        index = self.visit(node.slice)
        return PiSubscript(collection=collection, index=index)

    def visit_ClassDef(self, node: ast.ClassDef) -> PiClassDef:
        name = node.name
        methods = []
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                methods.append(self.visit_FunctionDef(stmt))
            else:
                raise ValueError("Seules les définitions de méthodes sont autorisées dans les classes.")
        return PiClassDef(name=name, methods=methods)

    def visit_Attribute(self, node: ast.Attribute) -> PiAttribute:
        obj = self.visit(node.value)
        attr = node.attr
        return PiAttribute(object=obj, attr=attr)

    def operator_symbol(self, op) -> str:
        if isinstance(op, ast.Add):
            return '+'
        elif isinstance(op, ast.Sub):
            return '-'
        elif isinstance(op, ast.Mult):
            return '*'
        elif isinstance(op, ast.Div):
            return '/'
        elif isinstance(op, ast.Mod):
            return '%'
        elif isinstance(op, ast.Eq):
            return '=='
        elif isinstance(op, ast.NotEq):
            return '!='
        elif isinstance(op, ast.Lt):
            return '<'
        elif isinstance(op, ast.LtE):
            return '<='
        elif isinstance(op, ast.Gt):
            return '>'
        elif isinstance(op, ast.GtE):
            return '>='
        elif isinstance(op, ast.In):
            return 'in'
        else:
            raise ValueError(f"Opérateur non pris en charge {op}.")

    def generic_visit(self, node):
        raise ValueError(f"Type de nœud AST non supporté : {type(node).__name__}")

