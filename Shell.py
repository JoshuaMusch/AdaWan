import Adawan
import sys

run = 1
if len(sys.argv) > 1:
    print("no Command line arguements Accepted")
    run = 0

while run:
    text = input('>>')

    if text.strip() == '': continue

    result, error = Adawan.Run('<stdin>', text)

    if error:
        print(error.AsString())
    elif result:
        if len(result.elements) == 1:
            print(repr(result.elements[0]))
        else:
            print(repr(result))
