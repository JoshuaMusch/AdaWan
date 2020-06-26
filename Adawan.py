import string

################################################################################
#                                   CONSTANTS                                  #
################################################################################

DIGITS         = '0123456789'
LETTERS        = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

################################################################################
#                                    ERRORS                                    #
################################################################################

class Error:
    def __init__ (self, posStart, posEnd, errorName, details):
        self.posStart  = posStart
        self.posEnd    = posEnd
        self.errorName = errorName
        self.details   = details

    def AsString (self):
        errorStr =  f'{self.errorName}: {self.details}\n'
        errorStr += f'File {self.posStart.fname}, line {self.posStart.line + 1}\n\n'
        errorStr += self.StringWithArrows()
        return errorStr

    def StringWithArrows (self):
        result    = ''
        idxStart  = max(self.posStart.ftxt.rfind('\n', 0, self.posStart.idx), 0)
        idxEnd    = self.posStart.ftxt.rfind('\n', idxStart + 1)
        if idxEnd < 0: idxEnd = len(self.posStart.ftxt)

        lineCount = self.posEnd.line - self.posStart.line + 1
        for i in range(lineCount):
            line     = self.posStart.ftxt[idxStart:idxEnd]
            colStart = self.posStart.col if i == 0 else 0
            colEnd   = self.posEnd.col if i == (lineCount - 1) else len(line) - 1

            result  += line + '\n'
            result  += ' ' * colStart + '^' * (colEnd - colStart)

            idxStart = idxEnd
            idxEnd   = self.posStart.ftxt.find('\n', idxStart + 1)
            if idxEnd < 0: idxEnd = len(self.posStart.ftxt)

        return result.replace('\t','')

class IllegalCharError(Error):
    def __init__ (self, posStart, posEnd, details):
        super().__init__(posStart, posEnd, 'Illegal Character', "'" + details + "'")

class InvalidSyntaxError(Error):
    def __init__ (self, posStart, posEnd, details=''):
        super().__init__(posStart, posEnd, 'Invalid Syntax', details)

class RunTimeError(Error):
    def __init__ (self, posStart, posEnd, details, context):
        super().__init__(posStart, posEnd, 'Runtime Error', details)
        self.context = context

    def AsString (self):
        errorStr  = self.GenerateTraceback()
        errorStr += f'{self.errorName}: {self.details}\n'
        errorStr += self.StringWithArrows()
        return errorStr

    def GenerateTraceback (self):
        result  = ''
        pos     = self.posStart
        context = self.context

        while context:
            result  = f'  File {pos.fname}, line {str(pos.line + 1)}, in {context.dispName}\n' + result
            pos     = context.parentEntryPos
            context = context.parent

        return 'Traceback (most Recent Call Last):\n' + result

################################################################################
#                                   POSITION                                   #
################################################################################

class Position:
    def __init__ (self, idx, line, col, fname, ftxt):
        self.idx   = idx
        self.line  = line
        self.col   = col
        self.fname = fname
        self.ftxt  = ftxt

    def Advance (self, currChar = None):
        self.idx  += 1
        self.col  += 1

        if currChar == '/n':
            self.line += 1
            self.col  =  0

        return self

    def Copy (self):
        return Position(self.idx, self.line, self.col, self.fname, self.ftxt)

################################################################################
#                                    TOKENS                                    #
################################################################################

TT_INT          = 'INT'
TT_FLOAT        = 'FLOAT'
TT_STRING		= 'STRING'
TT_IDENTIFIER	= 'IDENTIFIER'
TT_KEYWORD		= 'KEYWORD'

TT_PLUS     	= 'PLUS'
TT_MINUS    	= 'MINUS'
TT_MUL      	= 'MUL'
TT_DIV      	= 'DIV'

TT_POWER		= 'POWER'
TT_MOD          = 'MODULUS'
TT_FACTORIAL    = 'FACTORIAL'

TT_EQ			= 'EQ'

TT_LPAREN   	= 'LPAREN'
TT_RPAREN   	= 'RPAREN'

TT_EOF          = 'EOF'

KEYWORDS = [
    'LET'
]

class Token:
    def __init__ (self, _type, value=None, posStart = None, posEnd = None):
        self._type = _type
        self.value = value

        if posStart:
            self.posStart = posStart.Copy()
            self.posEnd   = posStart.Copy()
            self.posEnd.Advance()

        if posEnd:
            self.posEnd   = posEnd.Copy()

    def Matches (self, _type, value):
        return self._type == _type and self.value.upper() == value

    def __repr__ (self):
        if self.value:
            return f'{self._type}: {self.value}'
        else:
            return f'{self._type}'

################################################################################
#                                    Lexar                                     #
################################################################################

class Lexar:
    def __init__ (self, fname, text):
        self.fname    = fname
        self.text     = text
        self.currPos  = Position(-1, 0, -1, fname, text)
        self.currChar = None
        self.Advance()

    def Advance (self):
        self.currPos.Advance(self.currChar)
        if self.currPos.idx < len(self.text):
            self.currChar = self.text[self.currPos.idx]
        else:
            self.currChar = None

    def MakeTokens (self):
        tokens = [];

        while self.currChar != None:
            if self.currChar in ' \t':
                self.Advance()
            elif self.currChar in DIGITS:
                tokens.append(self.MakeNumber())
            elif self.currChar in LETTERS:
                tokens.append(self.MakeIdentifier())
            elif self.currChar == '+':
                tokens.append(Token(TT_PLUS,      posStart = self.currPos))
                self.Advance()
            elif self.currChar == '-':
                tokens.append(Token(TT_MINUS,     posStart = self.currPos))
                self.Advance()
            elif self.currChar == '*':
                tokens.append(Token(TT_MUL,       posStart = self.currPos))
                self.Advance()
            elif self.currChar == '/':
                tokens.append(Token(TT_DIV,       posStart = self.currPos))
                self.Advance()
            elif self.currChar == '%':
                tokens.append(Token(TT_MOD,       posStart = self.currPos))
                self.Advance()
            elif self.currChar == '^':
                tokens.append(Token(TT_POWER,     posStart = self.currPos))
                self.Advance()
            elif self.currChar == '!':
                poppedTokens = []
                currToken = tokens.pop()
                poppedTokens.append(currToken)

                if currToken._type == TT_RPAREN:
                    count_RPAREN = 1
                    while count_RPAREN > 0:
                        currToken = tokens.pop()
                        poppedTokens.append(currToken)
                        if currToken._type == TT_RPAREN: count_RPAREN += 1
                        if currToken._type == TT_LPAREN: count_RPAREN -= 1

                tokens.append(Token(TT_FACTORIAL, posStart = self.currPos))
                for x in range(len(poppedTokens)):
                    tokens.append(poppedTokens[len(poppedTokens)-x-1])
                self.Advance()
            elif self.currChar == '=':
                tokens.append(Token(TT_EQ,        posStart = self.currPos))
                self.Advance()
            elif self.currChar == '(':
                tokens.append(Token(TT_LPAREN,    posStart = self.currPos))
                self.Advance()
            elif self.currChar == ')':
                tokens.append(Token(TT_RPAREN,    posStart = self.currPos))
                self.Advance()
            else:
                posStart = self.currPos.Copy()
                char = self.currChar
                self.Advance()
                return [], IllegalCharError(posStart, self.currPos, char)

        tokens.append(Token(TT_EOF, posStart = self.currPos))

        return tokens, None

    def MakeNumber (self):
        numStr  = ''
        decimal = 0
        posStart = self.currPos.Copy()

        while self.currChar != None and self.currChar in DIGITS + '.':
            if self.currChar == '.':
                if decimal == 1: break
                decimal += 1
                numStr  += '.'
            else:
                numStr  += self.currChar

            self.Advance()

        if decimal == 0:
            return(Token(TT_INT,   int(numStr), posStart, self.currPos))
        else:
            return(Token(TT_FLOAT, float(numStr), posStart, self.currPos))

    def MakeIdentifier (self):
        idStr = ''
        posStart = self.currPos.Copy()

        while self.currChar != None and self.currChar in LETTERS_DIGITS + '_':
            idStr += self.currChar
            self.Advance()

        if idStr.upper() in KEYWORDS:
            tokType = TT_KEYWORD
        else:
            tokType = TT_IDENTIFIER

        return Token(tokType, idStr, posStart, self.currPos)

################################################################################
#                                    NODES                                     #
################################################################################

class NumberNode:
    def __init__ (self, tok):
        self.tok       = tok
        self.posStart  = self.tok.posStart
        self.posEnd    = self.tok.posEnd
    def __repr__ (self):
        return f'{self.tok}'

class BinOpNode:
    def __init__ (self, leftNode, opTok, rightNode):
        self.leftNode  = leftNode
        self.opTok     = opTok
        self.rightNode = rightNode
        self.posStart  = self.leftNode.posStart
        self.posEnd    = self.rightNode.posEnd
    def __repr__ (self):
        return f'({self.leftNode}, {self.opTok}, {self.rightNode})'

class UnaryOpNode:
    def __init__ (self, opTok, node):
        self.opTok     = opTok
        self.node      = node
        self.posStart  = self.opTok.posStart
        self.posEnd    = self.node.posEnd

    def __repr__ (self):
        return f'({self.opTok}, {self.node})'

class VarAssignNode:
    def __init__ (self, varNameTok, valueNode):
        self.varNameTok = varNameTok
        self.valueNode  = valueNode
        self.posStart   = self.varNameTok.posStart
        self.posEnd     = self.varNameTok.posEnd

class VarAccessNode:
    def __init__ (self, varNameTok):
        self.varNameTok = varNameTok
        self.posStart   = self.varNameTok.posStart
        self.posEnd     = self.varNameTok.posEnd


################################################################################
#                                   Parser                                     #
################################################################################

# Grammar - - - - - - - - - - - - - - - - - - -
# Expression : KEYWORD:LET IDENTIFIER EQ expr -
#            : term ((PLUS | MINUS) Term)*    -
#                                             -
# Term       : factor ((MUL|DIV|MOD) factor)* -
#                                             -
# Factor     : (PLUS | MINUS) Factor          -
#            : FACTORIAL Factor               -
#            : power                          -
#                                             -
# Power      : atom (POWER factor)*           -
#            : factorial
#                                             -
# Factorial  : FACTORIAL (atom)

# Atom       : INT | FLOAT | IDENTIFIER       -
#            : LPAREN expr RPAREN             -
# - - - - - - - - - - - - - - - - - - - - - - -

class ParseResult:
    def __init__ (self):
        self.error        = None
        self.node         = None
        self.advanceCount = 0

    def Register (self, res):
        self.advanceCount += res.advanceCount
        if res.error: self.error = res.error
        return res.node

    def RegisterAdvance (self):
        self.advanceCount += 1

    def Success (self, node):
        self.node = node
        return self

    def Failure (self, error):
        if not self.error or self.advanceCount == 0:
            self.error = error
        return self

class Parser:
    def __init__ (self, tokens):
        self.tokens  = tokens
        self.tok_idx = -1
        self.Advance()

    def Advance (self):
        self.tok_idx += 1
        if self.tok_idx < len(self.tokens):
            self.currToken = self.tokens[self.tok_idx]
        return self.currToken


    def Parse (self):
        res = self.Expr()
        if not res.error and self.currToken._type != TT_EOF:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected a mathematical operator"
            ))
        return res


    def Atom (self):
        res = ParseResult()
        tok = self.currToken
        if tok._type in (TT_INT, TT_FLOAT):
            res.RegisterAdvance()
            self.Advance()
            return res.Success(NumberNode(tok))

        elif tok._type == TT_IDENTIFIER:
            res.RegisterAdvance()
            self.Advance()
            return res.Success(VarAccessNode(tok))

        elif tok._type == TT_LPAREN:
            res.RegisterAdvance()
            self.Advance()
            expr = res.Register(self.Expr())
            if res.error: return res

            if self.currToken._type == TT_RPAREN:
                res.RegisterAdvance()
                self.Advance()
                return res.Success(expr)
            else:
                return res.Failure(InvalidSyntaxError(
                    self.currToken.posStart, self.currToken.posEnd,
                    "Expected ')'"
                ))

        return res.Failure(InvalidSyntaxError(
            tok.posStart, tok.posEnd,
            "Expected an Int | Float | Identifier | '+' | '-' | '('"
        ))

    def Power (self):
        return self.BinOp(self.Atom, (TT_POWER), self.Factor)

    def Factor (self):
        res = ParseResult()
        tok = self.currToken
        if tok._type in (TT_PLUS, TT_MINUS):
            res.RegisterAdvance()
            self.Advance()
            factor = res.Register(self.Factor())
            if res.error: return res
            return res.Success(UnaryOpNode(tok, factor))
        elif tok._type == TT_FACTORIAL:
            res.RegisterAdvance()
            self.Advance()
            factor = res.Register(self.Factor())
            if res.error: return res
            return res.Success(UnaryOpNode(tok, factor))

        return self.Power()

    def Term (self):
        return self.BinOp(self.Factor, (TT_MUL, TT_DIV, TT_MOD))

    def Expr (self):
        res = ParseResult()
        if self.currToken.Matches(TT_KEYWORD, 'LET'):
            res.RegisterAdvance()
            self.Advance()

            if self.currToken._type != TT_IDENTIFIER:
                return res.Failure(InvalidSyntaxError(
                    self.currToken.posStart, self.currToken.posEnd,
                    "Expected an Identifier"
                ))
            varName = self.currToken
            res.RegisterAdvance()
            self.Advance()

            if self.currToken._type != TT_EQ:
                return res.Failure(InvalidSyntaxError(
                    self.currToken.posStart, self.currToken.posEnd,
                    "Expected '='"
                ))

            res.RegisterAdvance()
            self.Advance()
            expr = res.Register(self.Expr())
            if res.error: return res
            return res.Success(VarAssignNode(varName, expr))

        node = res.Register(self.BinOp(self.Term, (TT_PLUS, TT_MINUS)))
        if res.error:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected an Int | Float | 'LET' | Identifier | '+' | '-' | '('"
            ))

        return res.Success(node)

    def BinOp (self, funcA, operations, funcB = None):
        if funcB == None: funcB = funcA
        res = ParseResult()
        left = res.Register(funcA())
        if res.error: return res

        while self.currToken._type in operations:
            opTok = self.currToken
            res.RegisterAdvance()
            self.Advance()
            right = res.Register(funcB())
            if res.error: return res
            left  = BinOpNode(left, opTok, right)

        return res.Success(left)

################################################################################
#                                RUNTIME RESULT                                #
################################################################################

class RTResult:
    def __init__ (self):
        self.value = None
        self.error = None

    def Register (self, res):
        if res.error: self.error = res.error
        return res.value

    def Success (self, value):
        self.value = value
        return self

    def Failure (self, error):
        self.error = error
        return self


################################################################################
#                                    VALUES                                    #
################################################################################

class Number:
    def __init__ (self, value):
        self.value = value
        self.SetPosition()
        self.SetContext(None)

    def SetPosition (self, posStart = None, posEnd = None):
        self.posStart = posStart
        self.posEnd   = posEnd
        return self

    def SetContext (self, context):
        self.context = context
        return self

    def AddedTo (self, B):
        if isinstance(B, Number):
            return Number(self.value + B.value).SetContext(self.context), None

    def SubtractedBy (self, B):
        if isinstance(B, Number):
            return Number(self.value - B.value).SetContext(self.context), None

    def MultipliedBy (self, B):
        if isinstance(B, Number):
            return Number(self.value * B.value).SetContext(self.context), None

    def Factorial (self):
        factorial = 1
        if self.value < 0:
            return None, RunTimeError(
                B.posStart, B.posEnd,
                "Factorial does not exist for negative numbers",
                self.context
            )
        else:
            for i in range(1, self.value + 1):
                factorial = factorial * i

        return Number(factorial).SetContext(self.context), None

    def PowerOf (self, B):
        if isinstance(B, Number):
            return Number(self.value ** B.value).SetContext(self.context), None

    def DividedBy (self, B):
        if isinstance(B, Number):
            if B.value == 0:
                return None, RunTimeError(
                    B.posStart, B.posEnd,
                    "Division By Zero",
                    self.context
                )
            return Number(self.value / B.value).SetContext(self.context), None

    def ModulusOf (self, B):
        if isinstance(B, Number):
            if B.value == 0:
                return None, RunTimeError(
                    B.posStart, B.posEnd,
                    "Division By Zero",
                    self.context
                )
            return Number(self.value % B.value).SetContext(self.context), None

    def Copy (self):
        copy = Number(self.value)
        copy.SetPosition(self.posStart, self.posEnd)
        copy.SetContext(self.context)
        return copy

    def __repr__ (self):
        return str(self.value)

################################################################################
#                                   CONTEXT                                    #
################################################################################

class Context:
    def __init__ (self, dispName, parent = None, parentEntryPos = None):
        self.dispName       = dispName
        self.parent         = parent
        self.parentEntryPos = parentEntryPos
        self.symbolTable    = None

################################################################################
#                                 SYMBOL TABLE                                 #
################################################################################

class SymbolTable:
    def __init__ (self):
        self.symbols = {}
        self.parent  = None # Global Variables

    def get (self, name):
        value = self.symbols.get(name, None)
        if value == None and self.parent:
            return self.parent.get(name)
        return value

    def set (self, name, value):
        self.symbols[name] = value

    def remove (self, name):
        del self.symbols[name]

################################################################################
#                                 INTERPRETER                                  #
################################################################################

class Interpreter:
    def Visit (self, node, context):
        methodName = f'visit_{type(node).__name__}'
        method     = getattr(self, methodName, self.NoVisitMethod)
        return method(node, context)

    def NoVisitMethod (self, node, context):
        raise exception(f'No visit_{type(node).__name__} method defined')

    def visit_NumberNode (self, node, context):
        return RTResult().Success(
            Number(node.tok.value).SetContext(context).SetPosition(node.posStart, node.posEnd)
        )

    def visit_VarAssignNode (self, node, context):
        res     = RTResult()
        varName = node.varNameTok.value
        value   = res.Register(self.Visit(node.valueNode, context))
        if res.error: return res

        context.symbolTable.set(varName, value)
        return res.Success(value)

    def visit_VarAccessNode (self, node, context):
        res     = RTResult()
        varName = node.varNameTok.value
        value   = context.symbolTable.get(varName)

        if not value:
            return res.Failure(RunTimeError(
                node.posStart, node.posEnd,
                f"'{varName}' is not defined",
                context
            ))

        value = value.Copy(). SetPosition(node.posStart, node.posEnd)
        return res.Success(value)

    def visit_BinOpNode (self, node, context):
        res   = RTResult()
        left  = res.Register(self.Visit(node.leftNode, context))
        if res.error: return res
        right = res.Register(self.Visit(node.rightNode, context))
        if res.error: return res

        error = None

        if node.opTok._type   == TT_PLUS:
            result, error = left.AddedTo(right)
        elif node.opTok._type == TT_MINUS:
            result, error = left.SubtractedBy(right)
        elif node.opTok._type == TT_MUL:
            result, error = left.MultipliedBy(right)
        elif node.opTok._type == TT_DIV:
            result, error = left.DividedBy(right)
        elif node.opTok._type == TT_MOD:
            result, error = left.ModulusOf(right)
        elif node.opTok._type == TT_POWER:
            result, error = left.PowerOf(right)

        if error:
            return res.Failure(error)
        else:
            return res.Success(result.SetPosition(node.posStart, node.posEnd))

    def visit_UnaryOpNode (self, node, context):
        res    = RTResult()
        number = res.Register(self.Visit(node.node, context))
        if res.error: return res

        error = None

        if node.opTok._type == TT_MINUS:
            number, error = number.MultipliedBy(Number(-1).SetContext(context))
        if node.opTok._type == TT_FACTORIAL:
            number, error = number.Factorial()

        if error:
            return res.Failure(error)
        else:
            return res.Success(number.SetPosition(node.posStart, node.posEnd))

################################################################################
#                                    Main                                      #
################################################################################

globalSymbolTable = SymbolTable()
globalSymbolTable.set("null", Number(0))

def Run (fname, text):
    # Generate Tokens
    lexar         = Lexar(fname, text)
    tokens, error = lexar.MakeTokens()
    if error: return None, error

    # Generate Abstract Search Tree
    parser = Parser(tokens)
    ast    = parser.Parse()
    if ast.error: return None, ast.error

    # Run the Interpreter
    interpreter = Interpreter()
    context     = Context('<Program>')
    context.symbolTable = globalSymbolTable
    res         = interpreter.Visit(ast.node, context)

    return res.value, res.error
