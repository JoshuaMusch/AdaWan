import Adawan
import sys

if len(sys.argv) == 2:
    debug = sys.argv[1]
    run   = 1
elif len(sys.argv) == 1:
    debug = 0
    run   = 1
else:
    print("no more than one command line input")
    run   = 0

while run:
    text = input('>>')

    if text == "exit": break
    if text.strip() == '': continue

    result, error = Adawan.Run('<stdin>', text)

    if error:
        print(error.AsString())
    elif result:
        if len(result.elements) == 1:
            print(repr(result.elements[0]))
        else:
            print(repr(result))
