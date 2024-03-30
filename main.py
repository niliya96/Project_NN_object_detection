def sum_digits(num):
    # Function to sum the digits of a number
    total = 0
    while num:
        total, num = total + num % 10, num // 10
    return total


def reduce_to_single_digit(num):
    # Function to reduce a number to a single digit
    while num > 9:
        num = sum_digits(num)
    return num


def aggregate_digits(numbers):
    # Function to aggregate the digits of a list of numbers
    sum_of_nums = sum(numbers)
    aggregated_sum = sum_digits(sum_of_nums)
    return reduce_to_single_digit(aggregated_sum)


def main():
    # List of 9-digit numbers
    numbers = [
        314880873,
        207921719
    ]

    # Aggregate the digits of the numbers in the list
    aggregated_result = aggregate_digits(numbers)

    print(f"Aggregated result of the numbers in the list is: {aggregated_result}")


if __name__ == "__main__":
    main()
