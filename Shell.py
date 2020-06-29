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
    if text != "exit":
        result, error = Adawan.Run('<stdin>', text, debug)

        if error:
            print(error.AsString())
        elif result:
            print(result)
    else:
        run = 0
