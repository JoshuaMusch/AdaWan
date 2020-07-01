import string
import os
import math

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

class ExpectedCharError(Error):
    def __init__ (self, posStart, posEnd, details):
        super().__init__(posStart, posEnd, 'Expected Character', details)

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

TT_EE           = 'EE'
TT_NE           = 'NE'
TT_LT           = 'LT'
TT_GT           = 'GT'
TT_LTE          = 'LTE'
TT_GTE          = 'GTE'

TT_COLON        = 'COLON'
TT_COMMA        = 'COMMA'

TT_EOF          = 'EOF'

KEYWORDS = [
    'let',
    'AND', '&',
    'OR',  '|',
    'NOT', '~',
    'if',
    'else',
    'elif',
    'for',
    'while',
    'function'
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
        return self._type == _type and self.value == value

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
            elif self.currChar == '"':
                tokens.append(self.MakeString())
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
            elif self.currChar == '(':
                tokens.append(Token(TT_LPAREN,    posStart = self.currPos))
                self.Advance()
            elif self.currChar == ')':
                tokens.append(Token(TT_RPAREN,    posStart = self.currPos))
                self.Advance()
            elif self.currChar == ',':
                tokens.append(Token(TT_COMMA,     posStart = self.currPos))
                self.Advance()
            elif self.currChar == '~':
                tok, error = self.MakeNotEquals()
                if error: return [], error
                tokens.append(tok)
            elif self.currChar == '=':
                tokens.append(self.MakeEquals())
            elif self.currChar == '<':
                tokens.append(self.MakeLessThan())
            elif self.currChar == '>':
                tokens.append(self.MakeGreaterThan())
            elif self.currChar == ':':
                tokens.append(Token(TT_COLON,     posStart = self.currPos))
                self.Advance()
            else:
                posStart = self.currPos.Copy()
                char = self.currChar
                self.Advance()
                return [], IllegalCharError(posStart, self.currPos, char)

        tokens.append(Token(TT_EOF,               posStart = self.currPos))

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
        idStr    = ''
        posStart = self.currPos.Copy()

        while self.currChar != None and self.currChar in LETTERS_DIGITS + '_':
            idStr += self.currChar
            self.Advance()

        if idStr in KEYWORDS:
            tokType = TT_KEYWORD
        else:
            tokType = TT_IDENTIFIER

        return Token(tokType, idStr, posStart, self.currPos)

    def MakeString (self):
        str        = ''
        posStart   = self.currPos.Copy()
        escapeChar = False
        self.Advance()

        escapeChars = {
			'n': '\n',
			't': '\t'
		}

        while self.currChar != None and (self.currChar != '"' or escapeChar):
            if escapeChar:

                str += escapeChars.get(self.currChar, self.currChar)
            else:
                if self.currChar == '\\':
                    escapeChar = True
                else:
                    str += self.currChar
                    self.Advance()

            escapeChar = False

        self.Advance()

        return Token(TT_STRING, str, posStart, self.currPos)

    def MakeNotEquals (self):
        posStart = self.currPos.Copy()
        self.Advance()

        if self.currChar == '=':
            self.Advance()
            return Token(TT_NE, posStart = posStart, posEnd = self.currPos), None

        self.Advance()
        return None, ExpectedCharError(posStart, self.currPos, "'=' (after '~')")

    def MakeEquals (self):
        tokType = TT_EQ
        posStart = self.currPos.Copy()
        self.Advance()

        if self.currChar == '=':
            self.Advance()
            tokType = TT_EE

        return Token(tokType, posStart = posStart, posEnd = self.currPos)

    def MakeLessThan (self):
        tokType = TT_LT
        posStart = self.currPos.Copy()
        self.Advance()

        if self.currChar == '=':
            self.Advance()
            tokType = TT_LTE

        return Token(tokType, posStart = posStart, posEnd = self.currPos)

    def MakeGreaterThan (self):
        tokType = TT_GT
        posStart = self.currPos.Copy()
        self.Advance()

        if self.currChar == '=':
            self.Advance()
            tokType = TT_GTE

        return Token(tokType, posStart = posStart, posEnd = self.currPos)

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

class StringNode:
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

class IfNode:
    def __init__ (self, cases, elseCase):
        self.cases    = cases
        self.elseCase = elseCase

        self.posStart = self.cases[0][0].posStart
        self.posEnd   = (self.elseCase or cases[len(cases) - 1][0]).posEnd

class ForNode:
    def __init__ (self, iteratorName, startNode, endNode, bodyNode, stepNode = None):
        self.iteratorName = iteratorName
        self.startNode    = startNode
        self.endNode      = endNode
        self.bodyNode     = bodyNode
        self.stepNode     = stepNode

        self.posStart     = iteratorName.posStart
        self.posEnd       = bodyNode.posEnd

class WhileNode:
    def __init__(self, conditionNode, bodyNode):
        self.conditionNode = conditionNode
        self.bodyNode      = bodyNode

        self.posStart      = conditionNode.posStart
        self.posEnd        = bodyNode.posEnd

    def __repr__ (self):
        return f'while: {self.conditionNode}'

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

class FuncDefNode:
    def __init__(self, funcNameTok, argNameToks, bodyNode):
        self.funcNameTok = funcNameTok
        self.argNameToks = argNameToks
        self.bodyNode    = bodyNode

        if len(argNameToks) > 0:
            self.posStart = argNameToks[0].posStart
        else:
            self.posStart = bodyNode.posStart

        self.posEnd = bodyNode.posEnd

class FuncCallNode:
    def __init__ (self, nodeToCall, argNodes):
        self.nodeToCall = nodeToCall
        self.argNodes   = argNodes

        self.posStart   = nodeToCall.posStart

        if len(argNodes) > 0:
            self.posEnd = argNodes[len(argNodes) - 1].posEnd
        else:
            self.posEnd = nodeToCall.posEnd

################################################################################
#                                   Parser                                     #
################################################################################

# Grammar - - - - - - - - - - - - - - - - - - - - - -
# Expression : KEYWORD:LET IDENTIFIER EQ expr       -
#            : comp-expr (AND|OR) comp-expr         -
#                                                   -
# Comp-expr  : NOT comp-expr                        -
#            : arith-expr (EE|NE) arith-expr        -
#                                                   -
# Arith-expr : term ((PLUS | MINUS) Term)*          -
#                                                   -
# Term       : factor ((MUL|DIV|MOD) factor)*       -
#                                                   -
# Factor     : (PLUS | MINUS) Factor                -
#            : FACTORIAL Factor                     -
#            : power                                -
#                                                   -
# Power      : call (POWER factor)*                 -
#            : factorial                            -
#                                                   -
# call       : atom (LPARN (expr(COMMA expr)*)?RPAREN)?
#                                                   -
# Atom       : INT | FLOAT | STRING | IDENTIFIER    -
#            : LPAREN expr RPAREN                   -
#            : If-expr                              -
#            : For-expr                             -
#            : While-expr                           -
#            : function-defn                        -
#                                                   -
# If-expr    : KW: IF expr KW: THEN expr            -
#            : (KW:ELIF expr KW: THEN expr)*        -
#            : (KW:ELSE expr KW: THEN expr)?        -
#                                                   -
# for-expr   : KW:FOR IDENTIFIER EQ expr KW:TO expr -
# 			 : (KW:STEP expr)? KW:THEN expr         -
#                                                   -
# while-expr : KW:WHILE expr KW:THEN expr           -
#                                                   -
# func-def   : KW:function IDENTIFIER               -
#                LPAREN (ID (COMMA ID)*)? RPAREN    -
#                COLON expr                         -
# - - - - - - - - - - - - - - - - - - - - - - - - - -

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
        if DEBUG: print(node)
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

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

    def Parse (self):
        res = self.Expr()
        if not res.error and self.currToken._type != TT_EOF:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected a mathematical operator"
            ))
        return res

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

    def FuncDef (self):
        res      = ParseResult()
        inputVar = []

        if not self.currToken.Matches(TT_KEYWORD, "function"):
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected 'function'"
            ))

        res.RegisterAdvance()
        self.Advance()

        if not self.currToken._type == TT_IDENTIFIER:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected an identifier"
            ))

        funcName = self.currToken
        res.RegisterAdvance()
        self.Advance()

        if not self.currToken._type == TT_LPAREN:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected an '('"
            ))

        res.RegisterAdvance()
        self.Advance()

        if self.currToken._type == TT_IDENTIFIER:
            inputVar.append(self.currToken)
            res.RegisterAdvance()
            self.Advance()

            while self.currToken._type == TT_COMMA:
                res.RegisterAdvance()
                self.Advance()

                if not self.currToken._type == TT_IDENTIFIER:
                    return res.Failure(InvalidSyntaxError(
                        self.currToken.posStart, self.currToken.posEnd,
                        "Expected an identifier"
                    ))

                inputVar.append(self.currToken)

                res.RegisterAdvance()
                self.Advance()

            if not self.currToken._type == TT_RPAREN:
                return res.Failure(InvalidSyntaxError(
                    self.currToken.posStart, self.currToken.posEnd,
                    "Expected a ',' | ')'"
                ))

        else:
            if not self.currToken._type == TT_RPAREN:
                return res.Failure(InvalidSyntaxError(
                    self.currToken.posStart, self.currToken.posEnd,
                    "Expected an identifier | ')'"
                ))

        res.RegisterAdvance()
        self.Advance()

        if not self.currToken._type == TT_COLON:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected ':'"
            ))

        res.RegisterAdvance()
        self.Advance()

        body = res.Register(self.Expr())
        if res.error: return res

        return res.Success(FuncDefNode(funcName, inputVar, body))

    def IfExpr (self):
        res      = ParseResult()
        cases    = []
        elseCase = None

        if not self.currToken.Matches(TT_KEYWORD, "if"):
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected 'if'"
            ))

        res.RegisterAdvance()
        self.Advance()

        condition = res.Register(self.Expr())
        if res.error: return res

        if not self.currToken._type == TT_COLON:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected ':'"
            ))

        res.RegisterAdvance()
        self.Advance()

        expr = res.Register(self.Expr())
        if res.error: return res
        cases.append((condition, expr))

        while self.currToken.Matches(TT_KEYWORD, "elif"):
            res.RegisterAdvance()
            self.Advance()

            condition = res.Register(self.Expr())
            if res.error: return res

            if not self.currToken._type == TT_COLON:
                return res.Failure(InvalidSyntaxError(
                    self.currToken.posStart, self.currToken.posEnd,
                    "Expected ':'"
                ))

            res.RegisterAdvance()
            self.Advance()

            expr = res.Register(self.Expr())
            if res.error: return res
            cases.append((condition, expr))

        if self.currToken.Matches(TT_KEYWORD,"else"):
            res.RegisterAdvance()
            self.Advance()

            elseCase = res.Register(self.Expr())
            if res.error: return res

        return res.Success(IfNode(cases, elseCase))

    def ForExpr (self):
        res = ParseResult()

        if not self.currToken.Matches(TT_KEYWORD, "for"):
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected 'for'"
            ))

        res.RegisterAdvance()
        self.Advance()

        if not self.currToken._type == TT_IDENTIFIER:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected an identifier"
            ))

        varName = self.currToken
        res.RegisterAdvance()
        self.Advance()

        if not self.currToken._type == TT_LPAREN:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected an '('"
            ))

        res.RegisterAdvance()
        self.Advance()

        startValue = res.Register(self.Expr())
        if res.error: return res

        if not self.currToken._type == TT_COLON:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected an ':'"
            ))

        res.RegisterAdvance()
        self.Advance()

        endValue = res.Register(self.Expr())
        if res.error: return res

        if self.currToken._type == TT_COLON:
            res.RegisterAdvance()
            self.Advance()

            stepValue = endValue
            endValue = res.Register(self.Expr())
            if res.error: return res
        else:
            stepValue = None

        if not self.currToken._type == TT_RPAREN:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected an ')'"
            ))

        res.RegisterAdvance()
        self.Advance()

        if not self.currToken._type == TT_COLON:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected ':'"
            ))

        res.RegisterAdvance()
        self.Advance()

        body = res.Register(self.Expr())
        if res.error: return res

        return res.Success(ForNode(varName, startValue, endValue, body, stepValue))

    def WhileExpr (self):
        res = ParseResult()

        if not self.currToken.Matches(TT_KEYWORD, "while"):
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected 'while'"
            ))

        res.RegisterAdvance()
        self.Advance()
        condition = res.Register(self.Expr())
        if res.error: return res

        if not self.currToken._type == TT_COLON:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected ':'"
            ))

        res.RegisterAdvance()
        self.Advance()

        body = res.Register(self.Expr())
        if res.error: return res

        return res.Success(WhileNode(condition, body))

    def Atom (self):
        res = ParseResult()
        tok = self.currToken
        if tok._type in (TT_INT, TT_FLOAT):
            res.RegisterAdvance()
            self.Advance()
            return res.Success(NumberNode(tok))

        elif tok._type == TT_STRING:
            res.RegisterAdvance()
            self.Advance()
            return res.Success(StringNode(tok))

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

        elif tok.Matches(TT_KEYWORD, "if"):
            ifExpr = res.Register(self.IfExpr())
            if res.error: return res

            return res.Success(ifExpr)

        elif tok.Matches(TT_KEYWORD, "for"):
            forExpr = res.Register(self.ForExpr())
            if res.error: return res

            return res.Success(forExpr)

        elif tok.Matches(TT_KEYWORD, "while"):
            WhileExpr = res.Register(self.WhileExpr())
            if res.error: return res

            return res.Success(WhileExpr)

        elif tok.Matches(TT_KEYWORD, "function"):
            funcDef = res.Register(self.FuncDef())
            if res.error: return res

            return res.Success(funcDef)

        return res.Failure(InvalidSyntaxError(
            tok.posStart, tok.posEnd,
            "Expected an Int | Float | Identifier | '+' | '-' | '(' | 'if' | 'for' | 'while' | 'function'"
        ))

    def Call (self):
        res = ParseResult()
        atom = res.Register(self.Atom())
        if res.error: return res

        if self.currToken._type == TT_LPAREN:
            res.RegisterAdvance()
            self.Advance()
            inputVar = []

            if self.currToken._type == TT_RPAREN:
                res.RegisterAdvance()
                self.Advance()
            else:
                inputVar.append(res.Register(self.Expr()))
                if res.error:
                    return res.Failure(InvalidSyntaxError(
                        self.currToken.posStart, self.currToken.posEnd,
                        "Expected a '(' | 'let' | 'if' | 'for' | 'while' | 'function' | int | float | identifier"
                    ))

                while self.currToken._type == TT_COMMA:
                    res.RegisterAdvance()
                    self.Advance()

                    inputVar.append(res.Register(self.Expr()))
                    if res.error: res

                if not self.currToken._type == TT_RPAREN:
                    return res.Failure(InvalidSyntaxError(
                        self.currToken.posStart, self.currToken.posEnd,
                        "Expected an ')'"
                    ))

                res.RegisterAdvance()
                self.Advance()
            return res.Success(FuncCallNode(atom, inputVar))
        return res.Success(atom)

    def Power (self):
        return self.BinOp(self.Call, (TT_POWER, ), self.Factor)

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

    def ArithExpr(self):
        return self.BinOp(self.Term, (TT_PLUS, TT_MINUS))

    def CompExpr (self):
        res = ParseResult()
        if self.currToken.Matches(TT_KEYWORD, 'NOT') or self.currToken.Matches(TT_KEYWORD, '~'):
            opTok = self.currToken
            res.RegisterAdvance()
            self.Advance()

            node = res.Register(self.CompExpr())
            if res.error: return res
            return res.Success(UnaryOpNode(opTok, node))

        node = res.Register(self.BinOp(self.ArithExpr, (TT_EE, TT_NE, TT_GT, TT_GTE, TT_LT, TT_LTE)))
        if res.error:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected an Int | Float | Identifier | '+' | '-' | '(' | 'NOT'"
            ))

        return res.Success(node)

    def Expr (self):
        res = ParseResult()
        if self.currToken.Matches(TT_KEYWORD, 'let'):
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

        node = res.Register(self.BinOp(self.CompExpr, ((TT_KEYWORD, 'AND'), (TT_KEYWORD, 'OR'))))
        # node = res.Register(self.BinOp(self.CompExpr,
        #     ((TT_KEYWORD, "AND"), (TT_KEYWORD, "&"),
        #     (TT_KEYWORD, "OR"),  (TT_KEYWORD, "|"))))
        if res.error:
            return res.Failure(InvalidSyntaxError(
                self.currToken.posStart, self.currToken.posEnd,
                "Expected an Int | Float | 'let' | Identifier | '+' | '-' | '(' | 'if' | 'for' | 'while' | 'function'"
            ))

        return res.Success(node)

    def BinOp (self, funcA, operations, funcB = None):
        if funcB == None: funcB = funcA
        res = ParseResult()
        left = res.Register(funcA())
        if res.error: return res

        while self.currToken._type in operations or (self.currToken._type, self.currToken.value) in operations:
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

class Value:
    def __init__ (self):
        self.SetPosition()
        self.SetContext()

    def SetPosition (self, posStart = None, posEnd = None):
        self.posStart = posStart
        self.posEnd   = posEnd
        return self

    def SetContext (self, context = None):
        self.context = context
        return self

    def AddedTo (self, B):
        return None, self.IllegalOperation(B)

    def SubtractedBy (self, B):
        return None, self.IllegalOperation(B)

    def MultipliedBy (self, B):
        return None, self.IllegalOperation(B)

    def Factorial (self):
        return None, self.IllegalOperation(B)

    def PowerOf (self, B):
        return None, self.IllegalOperation(B)

    def DividedBy (self, B):
        return None, self.IllegalOperation(B)

    def ModulusOf (self, B):
        return None, self.IllegalOperation(B)

    def GetComparison (self, tokType, B):
        return None, self.IllegalOperation(B)

    def AndedBy (self, B):
        return None, self.IllegalOperation(B)

    def OredBy (self, B):
        return None, self.IllegalOperation(B)

    def Notted (self):
        return None, self.IllegalOperation(B)

    def Execute (self):
        return None, self.IllegalOperation()

    def Copy (self):
        raise Exception('No copy method defined')

    def IsTrue (self):
        return False

    def IllegalOperation (self, B = None):
        if not B: B = self
        return RunTimeError(
            self.posStart, self.posEnd,
            "Illegal Operation",
            self.context
        )

class Number(Value):
    def __init__ (self, value):
        super().__init__()
        self.value = value

    def AddedTo (self, B):
        if isinstance(B, Number):
            return Number(self.value + B.value).SetContext(self.context), None
        else:
            return None, self.value.IllegalOperation(self.posStart, B.posEnd)

    def SubtractedBy (self, B):
        if isinstance(B, Number):
            return Number(self.value - B.value).SetContext(self.context), None
        else:
            return None, self.value.IllegalOperation(self.posStart, B.posEnd)

    def MultipliedBy (self, B):
        if isinstance(B, Number):
            return Number(self.value * B.value).SetContext(self.context), None
        else:
            return None, self.value.IllegalOperation(self.posStart, B.posEnd)

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
        else:
            return None, self.value.IllegalOperation(self.posStart, B.posEnd)

    def DividedBy (self, B):
        if isinstance(B, Number):
            if B.value == 0:
                return None, RunTimeError(
                    B.posStart, B.posEnd,
                    "Division By Zero",
                    self.context
                )
            return Number(self.value / B.value).SetContext(self.context), None
        else:
            return None, self.value.IllegalOperation(self.posStart, B.posEnd)

    def ModulusOf (self, B):
        if isinstance(B, Number):
            if B.value == 0:
                return None, RunTimeError(
                    B.posStart, B.posEnd,
                    "Division By Zero",
                    self.context
                )
            return Number(self.value % B.value).SetContext(self.context), None
        else:
            return None, self.value.IllegalOperation(self.posStart, B.posEnd)

    def GetComparison(self, tokType, B):
        if isinstance(B, Number):
            if tokType == TT_EE:
                return Number(int(self.value == B.value)).SetContext(self.context), None
            elif tokType == TT_NE:
                return Number(int(self.value != B.value)).SetContext(self.context), None
            elif tokType == TT_LT:
                return Number(int(self.value < B.value)).SetContext(self.context), None
            elif tokType == TT_GT:
                return Number(int(self.value > B.value)).SetContext(self.context), None
            elif tokType == TT_LTE:
                return Number(int(self.value <= B.value)).SetContext(self.context), None
            elif tokType == TT_GTE:
                return Number(int(self.value >= B.value)).SetContext(self.context), None
        else:
            return None, self.value.IllegalOperation(self.posStart, B.posEnd)

    def AndedBy(self, B):
        if isinstance(B, Number):
            return Number(int(self.value and B.value)).SetContext(self.context), None
        else:
            return None, self.value.IllegalOperation(self.posStart, B.posEnd)

    def OredBy(self, B):
        if isinstance(B, Number):
            return Number(int(self.value or B.value)).SetContext(self.context), None
        else:
            return None, self.value.IllegalOperation(self.posStart, B.posEnd)

    def Notted(self):
        return Number(1 if self.value == 0 else 0).SetContext(self.context), None

    def Copy (self):
        copy = Number(self.value)
        copy.SetPosition(self.posStart, self.posEnd)
        copy.SetContext(self.context)
        return copy

    def IsTrue(self):
        return self.value != 0

    def __repr__ (self):
        return str(self.value)

Number.null  = Number(   0   )
Number.false = Number(   0   )
Number.true  = Number(   1   )
Number.pi    = Number(math.pi)

class String(Value):
    def __init__ (self, value):
        super().__init__()
        self.value = value

    def AddedTo (self, B):
        if isinstance(B, String):
            return String(self.value + B.value).SetContext(self.context), None
        else:
            return None, IllegalOperation(self, B)

    def MultipliedBy (self, B):
        if isinstance(B, Number):
            return String(self.value * B.value).SetContext(self.context), None
        else:
            return None, IllegalOperation(self, B)

    def IsTrue (self):
        return len(self.value) > 0

    def Copy (self):
        copy = String(self.value)
        copy.SetContext(self.context)
        copy.SetPosition(self.posStart, self.posEnd)
        return copy

    def __str__ (self):
        return self.value

    def __repr__ (self):
        return f'"{self.value}"'

class BaseFunction(Value):
    def __init__ (self, name):
        self.name = name
        super().__init__()

    def GenerateNewContext (self):
        newContext = Context(self.name, self.context, self.posStart)
        newContext.symbolTable = SymbolTable(newContext.parent.symbolTable)
        return newContext

    def CheckArgs (self, argNames, args):
        res = RTResult()
        if len(args) != len(argNames):
            return res.Failure(RunTimeError(
                self.posStart, self.posEnd,
                f"Expected {len(argNames)} but recieved {len(args)}",
                self.context
            ))
        return res.Success(None)

    def PopulateArgs(self, argNames, args, exeContext):
        for i in range(len(args)):
            argName  = argNames[i]
            argValue = args[i]
            argValue.SetContext(exeContext)
            exeContext.symbolTable.set(argName, argValue)

    def CheckAndPopulateArgs(self, argNames, args, exeContext):
        res = RTResult()
        res.Register(self.CheckArgs(argNames, args))
        if res.error: return res
        self.PopulateArgs(argNames, args, exeContext)
        return res.Success(None)

class Function(BaseFunction):
    def __init__ (self, name, bodyNode, argNames):
        super().__init__(name)
        self.bodyNode = bodyNode
        self.argNames = argNames

    def Execute(self, args):
        res         = RTResult()
        interpreter = Interpreter()
        exeContext  = self.GenerateNewContext()

        res.Register(self.CheckAndPopulateArgs(self.argNames, args, exeContext))
        if res.error: return res

        value = res.Register(interpreter.Visit(self.bodyNode, exeContext))
        if res.error: return res

        return res.Success(value)

    def Copy (self):
        copy = Function(self.name, self.bodyNode, self.argNames)
        copy.SetContext(self.context)
        copy.SetPosition(self.posStart, self.posEnd)
        return copy

    def __repr__ (self):
        return f"<function {self.name}>"

class BuiltInFunction(BaseFunction):
    def __init__ (self, name):
        super().__init__(name)

    def Execute(self, args):
        res         = RTResult()
        exeContext  = self.GenerateNewContext()

        methodName = f'Execute_{self.name}'
        method     = getattr(self, methodName, self.NoVisitMethod)

        res.Register(self.CheckAndPopulateArgs(method.argNames, args, exeContext))
        if res.error: return res

        returnValue = res.Register(method(exeContext))
        if res.error: return res

        return res.Success(returnValue)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

    def NoVisitMethod (self, node, context):
        raise exception(f'No Execute_{type(node).__name__} method defined')

    def Execute_Print (self, exeContext):
        print(str(exeContext.symbolTable.get('value')))
        return RTResult().Success(Number.null)
    Execute_Print.argNames = ['value']

    def Execute_PrintReturn (self, exeContext):
        return RTResult().Success(String(str(exeContext.symbolTable.get('value'))))
    Execute_PrintReturn.argNames = ['value']

    def Execute_Input (self, exeContext):
        text = input()
        return RTResult().Sucess(String(text))
    Execute_Input.argNames = []

    def Execute_InputInt (self, exeContext):
        while True:
            text = input()
            try:
                number = int(text)
                break
            except ValueError:
                print(f"'{text}' must be an integer. Try again!")
        return RTResult().Success(Number(number))
    Execute_InputInt.argNames = []

    def Execute_Clear (self, exeContext):
        os.system('cls' if os.name == 'nt' else 'clear')
        return RTResult().Success(Number.null)
    Execute_Clear.argNames = []

    def Execute_IsNum (self, exeContext):
        isNum = isinstance(exeContext.symbolTable.get("value"), Number)
        return RTResult().Success(Number.true if isNum else Number.false)
    Execute_IsNum.argNames = ['value']

    def Execute_IsStr (self, exeContext):
        isStr = isinstance(exeContext.symbolTable.get("value"), String)
        return RTResult().Success(Number.true if isStr else Number.false)
    Execute_IsStr.argNames = ['value']

    def Execute_IsFunc (self, exeContext):
        isFunc = isinstance(exeContext.symbolTable.get("value"), BaseFunction)
        return RTResult().Success(Number.true if isFunc else Number.false)
    Execute_IsFunc.argNames = ['value']

    # def Execute_IsList (self, exeContext):
    #     isList = isinstance(exeContext.symbolTable.get("value"), List)
    #     return RTResult().Success(Number.true if isList else Number.false)
    # Exec_IsList.argNames = ['value']

    # def Execute_Append (self, exeContext):
    #     _list = exec_ctx.symbolTable.get("list")
    #     value = exec_ctx.symbolTable.get("value")
    #
    #     if not isinstance(_list, List):
    #         return RTResult().Failure(RunTimeError(
    #             self.posStart, self.posEnd,
    #             "First Argument must be a list",
    #             exeContext
    #         ))
    #
    #     _list.elements.append(value)
    #     return RTResult().Sucess(Number.null)
    # Execute_Append.argNames = ['list','value']

    # def Execute_Pop (self, exeContext):
    #     _list = exec_ctx.symbolTable.get("list")
    #     index = exec_ctx.symbolTable.get("index")
    #
    #     if not isinstance(_list, List):
    #         return RTResult().Failure(RunTimeError(
    #             self.posStart, self.posEnd,
    #             "the First Argument must be a list",
    #             exeContext
    #         ))
    #
    #     if not isinstance(index, Number):
    #         return RTResult().Failure(RunTimeError(
    #             self.posStart, self.posEnd,
    #             "the Second Argument must be a number",
    #             exeContext
    #         ))
    #
    #     try:
    #         element = _list.elements.pop(index.value)
    #     except:
    #         return RTResult().Failure(RunTimeError(
    #             self.posStart, self.posEnd,
    #             "Element could not be removed. The index is out of scope",
    #             exeContext
    #         ))
    #
    #     return RTResult().Sucess(element)
    # Execute_Pop.argNames = ['list','index']

    # def Execute_Pop (self, exeContext):
    #     _list = exec_ctx.symbolTable.get("listA")
    #     index = exec_ctx.symbolTable.get("listB")
    #
    #     if not isinstance(listA, List):
    #         return RTResult().Failure(RunTimeError(
    #             self.posStart, self.posEnd,
    #             "the First Argument must be a list",
    #             exeContext
    #         ))
    #
    #     if not isinstance(listB, List):
    #         return RTResult().Failure(RunTimeError(
    #             self.posStart, self.posEnd,
    #             "the Second Argument must be a list",
    #             exeContext
    #         ))
    #
    #     listA.elements.extend(listB.elements)
    #
    #     return RTResult().Sucess(Number.null)
    # Execute_Pop.argNames = ['listA','listB']

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

    def Copy (self):
        copy = BuiltInFunction(self.name)
        copy.SetContext(self.context)
        copy.SetPosition(self.posStart, self.posEnd)
        return copy

    def __repr__ (self):
        return f"<built-in function {self.name}>"

BuiltInFunction.Print       = BuiltInFunction("Print")
BuiltInFunction.PrintReturn = BuiltInFunction("PrintReturn")
BuiltInFunction.Input       = BuiltInFunction("Input")
BuiltInFunction.InputInt    = BuiltInFunction("InputInt")
BuiltInFunction.Clear       = BuiltInFunction("Clear")
BuiltInFunction.IsNum       = BuiltInFunction("IsNum")
BuiltInFunction.IsStr       = BuiltInFunction("IsStr")
BuiltInFunction.IsFunc      = BuiltInFunction("IsFunc")
# BuiltInFunction.IsList      = BuiltInFunction("IsList")
# BuiltInFunction.Append      = BuiltInFunction("Append")
# BuiltInFunction.Pop         = BuiltInFunction("Pop")
# BuiltInFunction.Extend      = BuiltInFunction("Extend")

################################################################################
#                                   CONTEXT                                    #
################################################################################

class Context:
    def __init__ (self, dispName, parent = None, parentEntryPos = None):
        self.dispName       = dispName
        self.parent         = parent
        self.parentEntryPos = parentEntryPos
        self.symbolTable    = None

    def __repr__ (self):
        return f"Name:{self.dispName}, Parent:({self.parent}: {self.parentEntryPos})"

################################################################################
#                                 SYMBOL TABLE                                 #
################################################################################

class SymbolTable:
    def __init__ (self, parent = None):
        self.symbols = {}
        self.parent  = parent

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
        methodName = f'Visit_{type(node).__name__}'
        method     = getattr(self, methodName, self.NoVisitMethod)
        return method(node, context)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

    def NoVisitMethod (self, node, context):
        raise exception(f'No visit_{type(node).__name__} method defined')

    def Visit_NumberNode (self, node, context):
        return RTResult().Success(
            Number(node.tok.value).SetContext(context).SetPosition(node.posStart, node.posEnd)
        )

    def Visit_StringNode (self, node, context):
        return RTResult().Success(
            String(node.tok.value).SetContext(context).SetPosition(node.posStart, node.posEnd)
        )

    def Visit_VarAssignNode (self, node, context):
        res     = RTResult()
        varName = node.varNameTok.value
        value   = res.Register(self.Visit(node.valueNode, context))
        if res.error: return res

        context.symbolTable.set(varName, value)
        return res.Success(value)

    def Visit_VarAccessNode (self, node, context):
        res     = RTResult()
        varName = node.varNameTok.value
        value   = context.symbolTable.get(varName)

        if not value:
            return res.Failure(RunTimeError(
                node.posStart, node.posEnd,
                f"'{varName}' is not defined",
                context
            ))

        value = value.Copy()
        value.SetContext(context)
        value.SetPosition(node.posStart, node.posEnd)
        return res.Success(value)

    def Visit_BinOpNode (self, node, context):
        res   = RTResult()
        left  = res.Register(self.Visit(node.leftNode, context))
        if res.error: return res
        right = res.Register(self.Visit(node.rightNode, context))
        if res.error: return res

        error = None
        tokType = node.opTok._type
        if tokType   == TT_PLUS:
            result, error = left.AddedTo(right)
        elif tokType == TT_MINUS:
            result, error = left.SubtractedBy(right)
        elif tokType == TT_MUL:
            result, error = left.MultipliedBy(right)
        elif tokType == TT_DIV:
            result, error = left.DividedBy(right)
        elif tokType == TT_MOD:
            result, error = left.ModulusOf(right)
        elif tokType == TT_POWER:
            result, error = left.PowerOf(right)
        elif tokType == TT_EE:
            result, error = left.GetComparison(TT_EE, right)
        elif tokType == TT_NE:
            result, error = left.GetComparison(TT_NE, right)
        elif tokType == TT_LT:
            result, error = left.GetComparison(TT_LT, right)
        elif tokType == TT_GT:
            result, error = left.GetComparison(TT_GT, right)
        elif tokType == TT_LTE:
            result, error = left.GetComparison(TT_LTE, right)
        elif tokType == TT_GTE:
            result, error = left.GetComparison(TT_GTE, right)
        elif node.opTok.Matches(TT_KEYWORD,'AND') or node.opTok.Matches(TT_KEYWORD,'&'):
            result, error = left.AndedBy(right)
        elif node.opTok.Matches(TT_KEYWORD,'OR')  or node.opTok.Matches(TT_KEYWORD,'|'):
            result, error = left.OredBy(right)

        if error:
            return res.Failure(error)
        else:
            return res.Success(result.SetPosition(node.posStart, node.posEnd))

    def Visit_UnaryOpNode (self, node, context):
        res    = RTResult()
        number = res.Register(self.Visit(node.node, context))
        if res.error: return res

        error = None

        if node.opTok._type   == TT_MINUS:
            number, error = number.MultipliedBy(Number(-1))
        elif node.opTok._type == TT_FACTORIAL:
            number, error = number.Factorial()
        elif node.opTok.Matches(TT_KEYWORD, 'NOT') or node.opTok.Matches(TT_KEYWORD, '~'):
            number, error = number.Notted()

        if error:
            return res.Failure(error)
        else:
            return res.Success(number.SetPosition(node.posStart, node.posEnd))

    def Visit_IfNode (self, node, context):
        res = RTResult()

        for condition, expr in node.cases:
            conditionValue = res.Register(self.Visit(condition, context))
            if res.error: return res

            if conditionValue.IsTrue():
                exprValue = res.Register(self.Visit(expr, context))
                if res.error: return res
                return res.Success(exprValue)

        if node.elseCase:
            elseValue = res.Register(self.Visit(node.elseCase, context))
            if res.error: return res
            return res.Success(elseValue)

        return res.Success(None)

    def Visit_ForNode (self, node, context):
        res = RTResult()
        startValue = res.Register(self.Visit(node.startNode, context))
        if res.error: return res

        if node.stepNode:
            stepValue = res.Register(self.Visit(node.stepNode, context))
            if res.error: return res
        else:
            stepValue = Number(1)

        endValue = res.Register(self.Visit(node.endNode, context))
        if res.error: return res

        i = startValue.value

        if stepValue.value >= 0:
            condition = lambda: i < endValue.value
        else:
            condition = lambda: i > endValue.value

        while condition():
            context.symbolTable.set(node.iteratorName.value, Number(i))
            i += stepValue.value
            res.Register(self.Visit(node.bodyNode, context))
            if res.error: return res

        return res.Success(None)

    def Visit_WhileNode (self, node, context):
        res = RTResult()
        while True:
            conditionValue = res.Register(self.Visit(node.conditionNode, context))
            if res.error: return res

            if not conditionValue.IsTrue(): break

            res.Register(self.Visit(node.bodyNode, context))
            if res.error: return res

        return res.Success(None)

    def Visit_FuncDefNode (self, node, context):
        res = RTResult()

        funcName = node.funcNameTok.value
        bodyNode = node.bodyNode
        argNames = [argName.value for argName in node.argNameToks]
        funcValue = Function(funcName, bodyNode, argNames)
        funcValue.SetContext(context)
        funcValue.SetPosition(node.posStart, node.posEnd)

        context.symbolTable.set(funcName, funcValue)

        return res.Success(funcValue)

    def Visit_FuncCallNode (self, node, context):
        res = RTResult()
        args = []

        valueToCall = res.Register(self.Visit(node.nodeToCall, context))
        if res.error: return res

        valueToCall = valueToCall.Copy().SetPosition(node.posStart, node.posEnd)

        for argNode in node.argNodes:
            args.append(res.Register(self.Visit(argNode, context)))
            if res.error: return res

        returnValue = res.Register(valueToCall.Execute(args))
        if res.error: return res

        returnValue = returnValue.Copy().SetPosition(node.posStart, node.posEnd).SetContext(context)
        if res.error: return res

        return res.Success(returnValue)

################################################################################
#                                    Main                                      #
################################################################################

DEBUG             = 0
globalSymbolTable = SymbolTable()
globalSymbolTable.set("Null",        Number.null                  )
globalSymbolTable.set("True",        Number.true                  )
globalSymbolTable.set("False",       Number.false                 )
globalSymbolTable.set("pi",          Number.pi                    )
globalSymbolTable.set("print",       BuiltInFunction.Print        )
globalSymbolTable.set("printReturn", BuiltInFunction.PrintReturn  )
globalSymbolTable.set("input",       BuiltInFunction.Input        )
globalSymbolTable.set("inputInt",    BuiltInFunction.InputInt     )
globalSymbolTable.set("clear",       BuiltInFunction.Clear        )
globalSymbolTable.set("isNum",       BuiltInFunction.IsNum        )
globalSymbolTable.set("isStr",       BuiltInFunction.IsStr        )
globalSymbolTable.set("isFunc",      BuiltInFunction.IsFunc       )
# globalSymbolTable.set("isList",      BuiltInFunction.IsList       )
# globalSymbolTable.set("append",      BuiltInFunction.Append       )
# globalSymbolTable.set("pop",         BuiltInFunction."Pop         )
# globalSymbolTable.set("extend",      BuiltInFunction.Extend       )

def Run (fname, text, debug = None):
    # if debug:
    #     DEBUG = 1
    # else:
    #     DEBUG = 0

    # Generate Tokens
    lexar         = Lexar(fname, text)
    tokens, error = lexar.MakeTokens()
    if error: return None, error

    if DEBUG: print("LEXAR:")
    if DEBUG: print(tokens)

    # Generate Abstract Search Tree
    parser = Parser(tokens)
    ast    = parser.Parse()
    if ast.error: return None, ast.error

    if DEBUG: print("PARSED:")
    if DEBUG: print(ast.node)

    # Run the Interpreter
    interpreter = Interpreter()
    context     = Context('<Program>')
    context.symbolTable = globalSymbolTable
    res         = interpreter.Visit(ast.node, context)

    return res.value, res.error
