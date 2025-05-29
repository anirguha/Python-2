# This is a decorator function that takes another function as an argument.
def passwordProtect(func):

    # This inner function is the one that will actually be called
    # when we use the decorator on another function.
    def wrappedFunc():
        password = input('Enter the password to call the function:')

        if password == 'password123': # correct password? then call the original function
            func()
        else: # If the password is not correct, deny access
            print("Access denied. Sorry, you need to enter the correct password to get the secret message.")

    return wrappedFunc


@passwordProtect
def printSecretMessage():
    secretMessage = "Shhh...this is a secret message"

    # We print a series of "~" characters the same length as the message,
    # then the message itself, then another series of "~" characters.
    print("~" * len(secretMessage))
    print(secretMessage)
    print("~" * len(secretMessage))

# By adding the decorator, we prompt the user for a password before printing the secret message.
# f = passwordProtect(printSecretMessage)
printSecretMessage()