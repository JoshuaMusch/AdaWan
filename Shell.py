import Adawan

while True:
    text = input('>>')
    result, error = Adawan.Run('<stdin>', text)

    if error:
        print(error.AsString())
    else:
        print(result)
