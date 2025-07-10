from dataclasses import dataclass

@dataclass
class PiNone:
    value: None

@dataclass
class PiNumber:
    value: float

@dataclass
class PiBool:
    value: bool

@dataclass
class PiVariable:
    name: str

@dataclass
class PiBinaryOperation:
    left: 'PiExpression'
    operator: str
    right: 'PiExpression'

@dataclass
class PiAssignment:
    name: str
    value: 'PiExpression'

@dataclass
class PiIfThenElse:
    condition: 'PiExpression'
    then_branch: list['PiStatement']
    else_branch: list['PiStatement']

@dataclass
class PiNot:
    operand: 'PiExpression'

@dataclass
class PiAnd:
    left: 'PiExpression'
    right: 'PiExpression'

@dataclass
class PiOr:
    left: 'PiExpression'
    right: 'PiExpression'

@dataclass
class PiWhile:
    condition: 'PiExpression'
    body: list['PiStatement']

@dataclass
class PiList:
    elements: list['PiExpression']

@dataclass
class PiTuple:
    elements: tuple['PiExpression', ...]

@dataclass
class PiString:
    value: str

@dataclass
class PiFunctionDef:
    name: str
    arg_names: list[str]
    vararg: str | None
    body: list['PiStatement']

@dataclass
class PiFunctionCall:
    function: 'PiExpression'
    args: list['PiExpression']

@dataclass
class PiFor:
    var: str
    iterable: 'PiExpression'
    body: list['PiStatement']

@dataclass
class PiBreak:
    pass

@dataclass
class PiContinue:
    pass

@dataclass
class PiIn:
    element: 'PiExpression'
    container: 'PiExpression'

@dataclass
class PiReturn:
    value: 'PiExpression'

@dataclass
class PiSubscript:
    collection: 'PiExpression'
    index: 'PiExpression'

@dataclass
class PiClassDef:
    name: str
    methods: list['PiFunctionDef']

@dataclass
class PiAttribute:
    object: 'PiExpression'
    attr: str

@dataclass
class PiAttributeAssignment:
    object: 'PiExpression'
    attr: str
    value: 'PiExpression'

PiValue = PiNumber | PiBool | PiNone | PiList | PiTuple | PiString

@dataclass
class PiInstanceCreation:
    class_name: str

@dataclass  
class PiAttributeAccess:
    class_name: str


PiExpression = (
    PiValue
    | PiVariable
    | PiBinaryOperation
    | PiNot
    | PiAnd
    | PiOr
    | PiFunctionCall
    | PiIn
    | PiSubscript
    | PiAttribute
    | PiAttributeAssignment
)

PiStatement = (
    PiAssignment
    | PiAttributeAssignment
    | PiIfThenElse
    | PiWhile
    | PiFor
    | PiBreak
    | PiContinue
    | PiFunctionDef
    | PiClassDef
    | PiReturn
    | PiExpression
)

PiProgram = list[PiStatement]
