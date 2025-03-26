# mymodule.py
import multiprocessing
import time


def greet(name):
    return f"Hello, {name}!"

PI = 3.1416




# -----------------------------------------------------------------------------#
# Multiprocessing
def calculate_square(numbers):
    for n in numbers:
        time.sleep(1)
        print(f"Square: {n ** 2}")

if __name__ == "__main__":  # Required for Windows
    numbers = [1, 2, 3, 4, 5]

    p1 = multiprocessing.Process(target=calculate_square, args=(numbers,))
    p2 = multiprocessing.Process(target=calculate_square, args=(numbers,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print("✅ Multiprocessing Done")
