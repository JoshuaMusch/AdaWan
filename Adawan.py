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
    def __init__ (self, posStart, posEnd, details):
        super().__init__(posStart, posEnd, 'Invalid Syntax', "'" + details + "'")

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
        self.tok = tok
    def __repr__ (self):
        return f'{self.tok}'

class BinOpNode:
    def __init__ (self, leftNode, opTok, rightNode):
        self.leftNode  = leftNode
        self.opTok     = opTok
        self.rightNode = rightNode
    def __repr__ (self):
        return f'({self.leftNode}, {self.opTok}, {self.rightNode})'

class UnaryOpNode:
    def __init__ (self, opTok, node):
        self.opTok = opTok
        self.node  = node

    def __repr__ (self):
        return f'({self.opTok}, {self.node})'

################################################################################
#                                   Parser                                     #
################################################################################

# Grammar - - - - - - - - - - - - - - - - - - -
# Expression : term ((PLUS | MINUS) Term)*    -
# Term       : factor ((MUL | DIV) factor)*   -
# Factor     : INT | FLOAT                    -
#            : (PLUS | MINUS) Factor          -
#            : FACTORIAL Factor               -
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
        print (self.tokens)
        res = self.Expr()
        if not res.error and self.currToken._type != 'EOF':
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected a mathematical operator"
            ))
        return res


    def Factor (self):
        res = ParseResult()
        tok = self.currToken
        print(tok._type)
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

        elif tok._type in ("INT", "FLOAT"):
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
            'Expected an Int or Float'
        ))

    def Term (self):
        return self.BinOp(self.Factor, ("MUL","DIV","POWER"))

    def Expr (self):
        return self.BinOp(self.Term, ("PLUS","MINUS"))


    def BinOp (self, func, operations):
        res = ParseResult()
        left = res.Register(func())
        if res.error: return res

        while self.currToken._type in operations:
            opTok = self.currToken
            res.Register(self.Advance())
            right = res.Register(func())
            if res.error: return res
            left  = BinOpNode(left, opTok, right)

        return res.Success(left)

################################################################################
#                                     Run                                      #
################################################################################

def Run (fname, text):
    lexar = Lexar(fname, text)
    tokens, error = lexar.MakeTokens()
    if error:
        return None, error

    # Generate Abstract Search Tree
    parser = Parser(tokens)
    ast    = parser.Parse()

    return ast.node, ast.error
