---
layout: post
title:  "some python tricks"
date:   2025-10-25 14:06:04 +0530
categories: tech
tokens: "~3.5k"
---

How to write more pythonic (god i hate this word) code

Random things I've learnt over the years which transform Python from "it works" to "it's beautiful"

---

## Function Magic: *args and **kwargs: Use them to your advantage

One of Python's most powerful features is its flexible argument handling. Instead of creating multiple function overloads, embrace `*args` and `**kwargs`.

Useful when you don't know how many arguments are going to be passed to your function at runtime.

```python
# ✅ Pythonic: Handle variable arguments elegantly
def add(*args):
    return sum(args)

print(add())            # 0
print(add(1, 2))        # 3
print(add(1, 2, 23, 5)) # 31 (works no matter how many arguments passed)

# For keyword arguments (dictionary like)
def printer(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} - {value}")

printer(language="python")
printer(name="samit", age=22) # name - samit, age - 22

```

**Why this matters:** This pattern eliminates the need for function overloading and makes your APIs incredibly flexible.

### Lambda Functions: Single expression functions

Helpful when you want to sort a sequence by some arbitrary computed key.

Lambda functions shine in sorting and functional programming contexts:

```python
# ✅ Elegant sorting with custom keys
tuples = [(1, "s"), (2, "a"), (3, "m"), (4, "i"), (5, "t")]

# suppose I want to sort with respect to alphabets(1st val) in this-:
sorted_tuples = sorted(tuples, key=lambda x: x[1])
# Result: [(2, 'a'), (4, 'i'), (3, 'm'), (1, 's'), (5, 't')]

# ✅ Replace switch statements with dictionaries
def dispatch(operator, x, y):
    return {
        "add": lambda: x + y,
        "sub": lambda: x - y,
        "mul": lambda: x * y,
        "div": lambda: x / y,
    }.get(operator, lambda: None)()

result = dispatch("mul", 2, 8)  # 16
```

**Functional programming patterns:**

```python
numbers = [1, 2, 3, 4, 5]

# ❌ Imperative style
squared = []
for num in numbers:
    squared.append(num ** 2)

# ✅ Functional style
squared = list(map(lambda n: n**2, numbers))

# ✅ Filtering
even_numbers = list(filter(lambda n: n % 2 == 0, numbers))
```


### LRU Cache: Dynamic Programming what??

#### Pre-requisite: Decorators

Decorators let you modify function behavior without changing the original code.

They let you do that without permanently modifying the wrapped function itself (the function behavior changes only when it's decorated).

Good for writing wrapper functions: modify behavior of callable through a wrapper closure so you don't have to permanently modify the original.

The original callable isn't modified → its behavior changes only when decorated.

```python
def uppercase(func):
    def wrapper():
        original_result = func()
        return original_result.upper()
    return wrapper

@uppercase
def greet():
    return "yo"

print(greet())  # "YO"
```

The `@lru_cache` decorator is like getting memoization for free:

```python
from functools import lru_cache

# ❌ Slow recursive Fibonacci
def fib_slow(n):
    if n <= 1:
        return n
    return fib_slow(n-1) + fib_slow(n-2)

# ✅ Fast cached version
@lru_cache(maxsize=None)
def fib_fast(n):
    if n <= 1:
        return n
    return fib_fast(n-1) + fib_fast(n-2)

print(fib_fast(40))  # Runs instantly instead of taking forever
# Just use lru_cache instead of figuring out DP forever.
```
---

### First-Class Functions and Higher-Order Magic

Python treats functions as first-class objects – you can pass them around like any other value. This opens up powerful programming patterns.

```python
def bark(text):
    return text.upper()

# Functions can be passed to other functions
result = list(map(bark, ["hello", "world"]))  # ['HELLO', 'WORLD']

# Essential for competitive programming (losers) input parsing
input_numbers = list(map(int, input("Enter numbers: ").split()))
# Input: "1 2 3 4 5" → Output: [1, 2, 3, 4, 5]
```

The `map()` function is great for transforming data efficiently. It's lazy (memory-efficient) and often more readable than list comprehensions for simple transformations. Really helpful when you're doing competitive programming (don't).

### More math stuff

```python
from itertools import permutations
from collections import Counter

# Quick permutations
list(permutations("ABC", 2))  # [('A', 'B'), ('A', 'C'), ('B', 'A'), ...]

# Frequency counting made trivial
arr = [1, 3, 4, 2, 1, 4, 1, 4, 2, 5, 2, 1, 4, 2, 1]
counter = Counter(arr) # Keeps number and frequency: {1 : 4, 2 : 4, 3 : 1} and so on
top_three = counter.most_common(3)
print(top_three)  # [(1, 5), (4, 4), (2, 4)]

# permutation&combinations 
numbers = [1, 2, 3, 4]
pairs = list(combinations(numbers, 2))  # [(1,2), (1,3), (1,4), (2,3), (2,4), (3,4)]
perm_pairs = list(permutations(numbers, 2))  # [(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (4, 1), (4, 2), (4, 3)]
```

### Don't use nested loops when you can use itertools

```python
from itertools import product, chain, combinations

# ❌ God this is bad
list_a = [1, 2020, 70]
list_b = [2, 4, 7, 2000]
list_c = [3, 70, 7]

for a in list_a:
    for b in list_b:
        for c in list_c:
            if a + b + c == 2077:
                print(a, b, c)  # 70 2000 7

# ✅ Hell yeah
list_a = [1, 2]
list_b = [3, 4]
for a, b in product(list_a, list_b):
    print(f"{a}, {b}")  # (1,3), (1,4), (2,3), (2,4)

```

### Essential Tools: zip & enumerate

```python
# enumerate for index + value (useful for leetcode problems where you need index as well)
for idx, val in enumerate(['a', 'b', 'c'], start=1):
    print(idx, val)
# 1 a
# 2 b
# 3 c

# zip to iterate in parallel
for x, y in zip([1, 2, 3], ['one', 'two', 'three']):
    print(x, y)
# 1 one
# 2 two
# 3 three
```

### Benchmarking: Measure Before You Optimize

```python
import time

# ❌ Bad code
numbers = [1, 2, 3, 4, 5]
sq_nums = []
square = lambda n: n**2

for num in numbers:
    sq_nums.append(square(num))

# ✅ Better
def sum_of_squares(n):
    """Calculate sum of squares from 1 to n."""
    return sum(i * i for i in range(1, n + 1))

# Performance measurement
start_time = time.perf_counter() # use perf_counter() instead of time.time()
result = sum_of_squares(100000)
end_time = time.perf_counter()

duration = end_time - start_time
print(f"Function took: {duration:.6f} seconds")  # Function took: 0.013979 seconds
```


### Generators: Lazy Evaluation for the Win

Generators are memory-efficient and elegant for processing large datasets.
No need to store the entire sequence in memory.

```python
# ❌ Memory-hungry approach
def read_large_file_bad(filename):
    with open(filename, 'r') as file:
        return file.read().split('\n')  # Loads entire file into memory

# ✅ Memory-efficient generator
def read_large_file_good(filename):
    with open(filename, 'r') as file:
        for line in file:
            yield line.strip()  # does not load the entire file, just prints

# ✅ Infinite sequences → since computer has finite memory we use yield
def inf_numbers():
    for i in range(1000000000000000000000000000):
        # print(i) # BAD
        yield i # much faster

# Generator expressions for memory optimised, much faster code
squares_list = [x**2 for x in range(1000000)]      # Size ~37MB
squares_generator = (x**2 for x in range(1000000)) # Size ~88 bytes
```

### Shallow vs Deep Copy

Understanding shallow vs deep copying can save you from subtle bugs:

```python
import copy

# ❌ Shallow copy trap
xs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
ys = list(xs)  # Shallow copy
xs[1][0] = "X"
print(ys)  # ys is also affected! 

# ✅ Deep copy solution → no reference to original, makes independent copy
xs = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
zs = copy.deepcopy(xs)
xs[1][0] = "Y"  # [[1, 2, 3], ['Y', 5, 6], [7, 8, 9]]
print(zs)       # zs remains unchanged ✅ [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

### Named Tuples: Classes Without the Boilerplate

Memory efficient shortcut to defining an immutable class in Python manually.

```python
from collections import namedtuple

# ✅ Clean, immutable data structures
Car = namedtuple("Car", ["color", "mileage"])
my_car = Car("red", 1000)
print(my_car.color)  # Accessible like attributes
```

### Static Methods: Clear Intent

Static methods can't access class or instance state because they don't take a `cls` or `self` argument.

It's a great signal to show that particular method is independent from everything else around it.

```python
import math

class Pizza:
    def __init__(self, radius, ingredients):
        self.radius = radius
        self.ingredients = ingredients
    
    def area(self):
        return self.circle_area(self.radius)
    
    @staticmethod
    def circle_area(r):
        """Independent utility function"""
        return math.pi * r**2

# Can be called without instance
area = Pizza.circle_area(5)
```

[More on Object Oriented Programming in Python](tab:https://github.com/samitmohan/interviews/tree/master/PythonLanguageSpecific/OOPs)

### Dictionary Merging: The Modern Way

```python
# ✅ With unpacking
xs = {"a": 1, "b": 2}
ys = {"c": 3, "d": 4}
merged = {**xs, **ys}  # {'a': 1, 'b': 2, 'c': 3, 'd': 4}
```

### Matrix Operations: Elegant Transformations

```python
# ✅ Transpose matrix with zip (so clean)
matrix = [[8, 9, 10], [11, 12, 13]]
transposed = list(zip(*matrix))  # [(8, 11), (9, 12), (10, 13)]

# ✅ Flatten nested lists
import itertools
nested = [[1, 2], [3, 4], [5, 6]]
flattened = list(itertools.chain.from_iterable(nested))  # [1, 2, 3, 4, 5, 6]
```

### DefaultDict: Eliminate Key Checking

```python
from collections import defaultdict

# ❌ Manual key checking
word_count = {}
for word in ["apple", "banana", "apple"]:
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1

# ✅ DefaultDict elegance
word_count = defaultdict(int)
for word in ["apple", "banana", "apple"]:
    word_count[word] += 1  # apple: 2, banana: 1

# Similarly counter function calculates frequency of word → very useful for solving leetcode questions
from collections import Counter
a = ["samit", "samit", "mohan"]
hm = Counter(a)  # Counter({'samit': 2, 'mohan': 1})
list(hm.values())  # [2, 1]
```

### Binary Search with Bisect

When to use Bisect? Searching in sorted arrays or maintaing sorted collections.

```python
import bisect

# Use binary search O(log n)
def binary_search(arr, target):
    index = bisect.bisect_left(arr, target)
    return index if index < len(arr) and arr[index] == target else -1

arr = [1, 3, 5, 7, 9]

# bisect_left: Returns leftmost insertion point
# If element exists, returns its index
# If element doesn't exist, returns where it should be inserted
pos_left = bisect.bisect_left(arr, 5)    # Returns 2 (index of 5)
pos_left = bisect.bisect_left(arr, 6)    # Returns 3 (between 5 and 7)

# bisect_right (or just bisect): Returns rightmost insertion point
# If element exists, returns index AFTER the last occurrence
pos_right = bisect.bisect_right(arr, 5)  # Returns 3 (after the 5)
pos_right = bisect.bisect_right(arr, 6)  # Returns 3 (same as left for non-existent)

# What is the difference between bisect_left and bisect_right: For arrays with duplicates, the difference matters:
arr_with_dups = [1, 3, 5, 5, 5, 7, 9]
left_pos = bisect.bisect_left(arr_with_dups, 5)   # Returns 2 (first 5)
right_pos = bisect.bisect_right(arr_with_dups, 5) # Returns 5 (after last 5)

# insort_left and insort_right: Insert and maintain sorted order O(n) for insertion, O(log n) for finding
bisect.insort_left(arr, 6)   # arr becomes [1, 3, 5, 6, 7, 9]
bisect.insort_right(arr, 4)  # arr becomes [1, 3, 4, 5, 6, 7, 9]
```

### Use Type Annotations

Always use type annotations, always use docstrings under functions. Know what is expected to be the input and output of your function.

```python
from typing import List, Dict

# ❌ Unclear function signature
def calculate_total(items, discount):
    pass

# Readability is quite shit, what are items? is it a list, tuple, dictionary? 
# discount might be an integer (guessing)

# ✅ Use Types
def calculate_total_price( items: List[Dict[str, float]], discount_rate: float) -> float:
    """
    Calculate total price with discount applied.
    
    Args:
        items: List of item dictionaries with 'price' and 'quantity'
        discount_rate: Discount as decimal (0.1 = 10%)
    
    Returns:
        Total discounted price
    """
    total = sum(item['price'] * item['quantity'] for item in items)
    return total * (1 - discount_rate)
```

### Keep It Simple

Please don't write code like this:

```python
def fib(x):
    if x <= 1: 
        return x
    else:
        return fib(x - 1) + fib(x - 2)
```

when you can write it like this:

```python
def fib(x):
    return x if x <= 1 else fib(x - 1) + fib(x - 2)
```

### String Formatting: F-Strings Rule

```python
name, age = "Samit", 22

# ❌ Old
message = "My name is " + name + " and I am " + str(age) + " years old"

# ✅ Formatting (cleaner)
print(f"My name is {name} and I'm {age} yrs old")
```

### Context Managers: Resource Management

```python
# ❌ Manual resource management (don't do this cmon it's 2025)
file = open("data.txt", "w")
try:
    file.write("Hello World")
finally:
    file.close()

# ✅ Much better
with open("data.txt", "w") as file:
    file.write("Hello World")
```

### The Walrus Operator: Assignment in Expressions

```python
# ✅ Assign and use in one expression
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
if (length := len(data)) > 10:
    print(f"List is long: {length} items")

# ✅ Used for efficient input processing
while (command := input("$ ")) != "exit":
    os.system(command)
```

### Unpacking and Destructuring

```python
# ✅ Merge different iterables
list_a = [1, 2, 3]
tuple_b = (4, 5, 6)
set_c = {7, 8, 9}
merged = [*list_a, *tuple_b, *set_c]  # [1, 2, 3, 4, 5, 6, 8, 9, 7]

# ✅ Function argument unpacking
def greet(first, last, age):
    return f"Hello {first} {last}, age {age}"

person_data = ("John", "Doe", 30)
greeting = greet(*person_data) # unpacks tuples and passes to function: Hello John Doe, age 30
```

### Boolean Logic Shortcuts

```python
numbers = [2, 4, 6, 8, 10]

# Check if all elements satisfy condition
all_even = all(n % 2 == 0 for n in numbers)  # True

# Check if any element satisfies condition  
any_greater_than_5 = any(n > 5 for n in numbers)  # True
```

### Palindrome Check: Slicing Magic

```python
def is_palindrome(text: str) -> bool:
    return text == text[::-1]

print(is_palindrome("racecar"))  # True

numbers[::2]    # Every second element
numbers[::-1]   # Reverse
numbers[1::2]   # Skip first, then every second
```

### Package Management

Stop using `pip` directly for project management. Use `uv` instead:

```bash
# ❌ Old way
pip install requests
pip freeze > requirements.txt

# ✅ Modern way with uv (written in Rust, blazingly fast)
uv add requests
uv sync
```

- [uv Documentation](https://github.com/astral-sh/uv)

### Performance Insights: Under the Hood

Understanding Python's internals helps you write faster code.

The `dis` module shows Python bytecode - the internal operations.

```python
import dis

# Tuples are compiled more efficiently
dis.dis(compile("(28, 's', 'a', 'm')", "", "eval"))
# Output: Simple RETURN_CONST operation
#   0           RESUME                   0
#   1           RETURN_CONST             0 ((28, 's', 'a', 'm'))

dis.dis(compile("[28, 's', 'a', 'm']", "", "eval"))
# Output: Multiple operations including BUILD_LIST
#   0           RESUME                   0
#   1           BUILD_LIST               0
#               LOAD_CONST               0 ((28, 's', 'a', 'm'))
#               LIST_EXTEND              1
#               RETURN_VALUE
```

**Key takeaway:** Tuples are pre-computed at compile time, lists are built at runtime.

---

## Conclusion

Zen of Python:

```python
>>> import this
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

---
