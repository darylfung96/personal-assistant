import time
from functools import wraps


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Call the function
        end_time = time.time()  # Record end time
        runtime = end_time - start_time  # Calculate runtime
        print(f"Runtime of {func.__name__}: {runtime:.4f} seconds")
        return result  # Return the result of the function

    return wrapper
