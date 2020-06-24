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

    def AsString(self):
        errorStr =  f'{self.errorName}: {self.details}\n'
        errorStr += f'File {self.posStart.fname}, line {self.posStart.line + 1}\n'
        return errorStr

class IllegalCharError(Error):
    def __init__ (self, posStart, posEnd, details):
        super().__init__(posStart, posEnd, 'Illegal Character', "'" + details + "'")

################################################################################
#                                   POSITION                                   #
################################################################################

class Position:
    def __init__(self, idx, line, col, fname, ftxt):
        self.idx   = idx
        self.line  = line
        self.col   = col
        self.fname = fname
        self.ftxt  = ftxt

    def Advance(self, currChar):
        self.idx  += 1
        self.col  += 1

        if currChar == '/n':
            self.line += 1
            self.col  =  0

        return self

    def Copy(self):
        return Position(self.idx, self.line, self.col, self.fname, self.ftxt)

################################################################################
#                                    TOKENS                                    #
################################################################################

class Token:
    def __init__ (self, _type, value=None):
        self._type  = _type
        self.value = value
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
                tokens.append(Token('PLUS'))
                self.Advance()
            elif self.currChar == '-':
                tokens.append(Token('MINUS'))
                self.Advance()
            elif self.currChar == '*':
                tokens.append(Token('MUL'))
                self.Advance()
            elif self.currChar == '/':
                tokens.append(Token('DIV'))
                self.Advance()
            elif self.currChar == '%':
                tokens.append(Token('MOD'))
                self.Advance()
            elif self.currChar == '!':
                tokens.append(Token('FACTORIAL'))
                self.Advance()
            elif self.currChar == '(':
                tokens.append(Token('LPAREN'))
                self.Advance()
            elif self.currChar == ')':
                tokens.append(Token('RPAREN'))
                self.Advance()
            else:
                posStart = self.currPos.Copy()
                char = self.currChar
                self.Advance()
                return [], IllegalCharError(posStart, self.currPos, char)
        return tokens, None

    def MakeNumber (self):
        numStr  = ''
        decimal = 0

        while self.currChar != None and self.currChar in DIGITS + '.':
            if self.currChar == '.':
                if decimal == 1: break
                decimal += 1
                numStr  += '.'
            else:
                numStr  += self.currChar

            self.Advance()

        if decimal == 0:
            return(Token('INT',   int(numStr)))
        else:
            return(Token('FLOAT', float(numStr)))

################################################################################
#                                     Run                                      #
################################################################################

def Run(fname, text):
    lexar = Lexar(fname, text)
    tokens, error = lexar.MakeTokens()

    return tokens, error
