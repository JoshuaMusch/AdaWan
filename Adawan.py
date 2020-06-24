################################################################################
#                                   CONSTANTS                                  #
################################################################################

DIGITS = '0123456789'

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

    def StringWithArrows(self):
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
            elif self.currChar == '+':
                tokens.append(Token('PLUS',      posStart = self.currPos))
                self.Advance()
            elif self.currChar == '-':
                tokens.append(Token('MINUS',     posStart = self.currPos))
                self.Advance()
            elif self.currChar == '*':
                tokens.append(Token('MUL',       posStart = self.currPos))
                self.Advance()
            elif self.currChar == '/':
                tokens.append(Token('DIV',       posStart = self.currPos))
                self.Advance()
            elif self.currChar == '%':
                tokens.append(Token('MOD',       posStart = self.currPos))
                self.Advance()
            elif self.currChar == '!':
                tokens.append(Token('FACTORIAL', posStart = self.currPos))
                self.Advance()
            elif self.currChar == '^':
                tokens.append(Token('POWER',     posStart = self.currPos))
                self.Advance()
            elif self.currChar == '(':
                tokens.append(Token('LPAREN',    posStart = self.currPos))
                self.Advance()
            elif self.currChar == ')':
                tokens.append(Token('RPAREN',    posStart = self.currPos))
                self.Advance()
            else:
                posStart = self.currPos.Copy()
                char = self.currChar
                self.Advance()
                return [], IllegalCharError(posStart, self.currPos, char)

        tokens.append(Token('EOF', posStart = self.currPos))

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
            return(Token('INT',   int(numStr), posStart, self.currPos))
        else:
            return(Token('FLOAT', float(numStr), posStart, self.currPos))

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

################################################################################
#                                   Parser                                     #
################################################################################

# Grammar - - - - - - - - - - - - - - - - - - -
# Expression : term ((PLUS | MINUS) Term)*    -
#                                             -
# Term       : factor ((MUL | DIV) factor)*   -
#                                             -
# Factor     : (PLUS | MINUS) Factor          -
#            : power                          -
#                                             -
# Power      : atom (POWER factor)*           -
#            : FACTORIAL Factor               -
#                                             -
# Atom       : INT | FLOAT                    -
#            : LPAREN expr RPAREN             -
# - - - - - - - - - - - - - - - - - - - - - - -

class ParseResult:
    def __init__ (self):
        self.error = None
        self.node  = None

    def Register (self, res):
        if isinstance(res, ParseResult):
            if res.error:
                self.error = res.error
            return res.node
        return res

    def Success (self, node):
        self.node = node
        return self

    def Failure (self, error):
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
        if not res.error and self.currToken._type != 'EOF':
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected a mathematical operator"
            ))
        return res

    def Atom (self):
        res = ParseResult()
        tok = self.currToken
        if tok._type in ("INT", "FLOAT"):
            res.Register(self.Advance())
            return res.Success(NumberNode(tok))

        elif tok._type == "LPAREN":
            res.Register(self.Advance())
            expr = res.Register(self.Expr())

            if res.error: return res
            if self.currToken._type == "RPAREN":
                res.Register(self.Advance())
                return res.Success(expr)
            else:
                return res.Failure(InvalidSyntaxError(
                    self.currToken.posStart, self.currToken.posEnd,
                    "Expected ')'"
                ))
        return res.Failure(InvalidSyntaxError(
            tok.posStart, tok.posEnd,
            "Expected an Int | Float | '+' | '-' | '('"
        ))

    def Power(self):
        return self.BinOp(self.Atom, ("POWER"), self.Factor)

    def Factor (self):
        res = ParseResult()
        tok = self.currToken
        if tok._type == "FACTORIAL":
            res.Register(self.Advance())
            factor = res.Register(self.Factor())
            if res.error: return res
            return res.Success(UnaryOpNode(tok, factor))

        elif tok._type in ("PLUS", "MINUS"):
            res.Register(self.Advance())
            factor = res.Register(self.Factor())
            if res.error: return res
            return res.Success(UnaryOpNode(tok, factor))

        return self.Power()

    def Term (self):
        return self.BinOp(self.Factor, ("MUL","DIV"))

    def Expr (self):
        return self.BinOp(self.Term, ("PLUS","MINUS"))


    def BinOp (self, funcA, operations, funcB = None):
        if funcB == None: funcB = funcA
        res = ParseResult()
        left = res.Register(funcA())
        if res.error: return res

        while self.currToken._type in operations:
            opTok = self.currToken
            res.Register(self.Advance())
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

    def SetContext(self, context):
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

    def DividedBy (self, B):
        if isinstance(B, Number):
            if B.value == 0:
                return None, RunTimeError(
                    B.posStart, B.posEnd,
                    "Division By Zero",
                    self.context
                )
            return Number(self.value / B.value).SetContext(self.context), None

    def PowerOf (self, B):
        if isinstance(B, Number):
            return Number(self.value ** B.value).SetContext(self.context), None

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

    def visit_BinOpNode (self, node, context):
        res   = RTResult()
        left  = res.Register(self.Visit(node.leftNode, context))
        if res.error: return res
        right = res.Register(self.Visit(node.rightNode, context))
        if res.error: return res

        error = None

        if node.opTok._type == "PLUS":
            result, error = left.AddedTo(right)
        elif node.opTok._type == "MINUS":
            result, error = left.SubtractedBy(right)
        elif node.opTok._type == "MUL":
            result, error = left.MultipliedBy(right)
        elif node.opTok._type == "DIV":
            result, error = left.DividedBy(right)
        elif node.opTok._type == "POWER":
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

        if node.opTok._type == "MINUS":
            number, error = number.MultipliedBy(Number(-1).SetContext(context))

        if error:
            return res.Failure(error)
        else:
            return res.Success(number.SetPosition(node.posStart, node.posEnd))

################################################################################
#                                    Main                                      #
################################################################################

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
    res         = interpreter.Visit(ast.node, context)

    return res.value, res.error
