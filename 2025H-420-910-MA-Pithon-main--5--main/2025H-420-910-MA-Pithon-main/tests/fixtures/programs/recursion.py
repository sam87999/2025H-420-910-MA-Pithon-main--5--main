# Test de Alexandrine Dubé
def factorial(n):
    # Base case: The condition that stops the recursion
    if n == 0 or n == 1:
        return 1
    # Recursive case: The function calls itself with a modified argument
    else:
        to_send = n - 1
        new_val = factorial(to_send)
        mine = n * new_val
        return mine


result = factorial(5)
print(result) # Output: 120
print(factorial(10)) # Output: 3628800
print(factorial(34)) # Output: 295232799039604140847618609643520000000
print(factorial(0)) # Output: 1