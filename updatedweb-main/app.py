import timeit
import traceback
from flask import Flask, request, render_template, jsonify
from exponential_search import exponential_search, exponential_search_wrapper
from binary_search import binary_search, binary_search_wrapper
from interpolation_search import interpolation_search, interpolation_search_wrapper
from jump_search import jump_search, jump_search_wrapper
from linear_search import linear_search, linear_search_wrapper
from ternary_search import ternary_search, ternary_search_wrapper
from linked_list import Queue as MyQueue, Deque as MyDeque

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/queue")
def queue():
    return render_template("queue.html")

@app.route("/dequeu")
def dequeu():
    return render_template("dequeu.html")

def is_operand(token):
    return token.isalnum()

def get_precedence(operator):
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    return precedence.get(operator, 0)

def infix_to_postfix(infix_expression):
    infix_expression = infix_expression.replace(" ", "")  
    ops_stack = []
    output_stack = []
    postfix_steps = []

    for token in infix_expression:
        if is_operand(token):
            output_stack.append(token)
        elif token == '(':
            ops_stack.append(token)
        elif token == ')':
            while ops_stack and ops_stack[-1] != '(':
                output_stack.append(ops_stack.pop())
            ops_stack.pop()
        elif token in {'+', '-', '*', '/'}:
            while ops_stack and get_precedence(ops_stack[-1]) >= get_precedence(token):
                output_stack.append(ops_stack.pop())

            ops_stack.append(token)

        postfix_result = ' '.join(output_stack)
        postfix_steps.append(f'{postfix_result}')

    while ops_stack:
        output_stack.append(ops_stack.pop())

    postfix_result = ' '.join(output_stack)
    postfix_steps.append(f'{postfix_result}')

    return postfix_steps

@app.route('/Stacks', methods=['GET', 'POST'])
def Stacks():
    if request.method == 'POST':
        infix_expression = request.form['infix_expression']
        postfix_steps = infix_to_postfix(infix_expression)
        return render_template('Stacks.html', infix_expression=infix_expression, postfix_steps=postfix_steps)
    return render_template('Stacks.html')

@app.route("/Profpage")
def Profpage():
    return render_template("Profpage.html")

def generate_array(dataset_size):
    return list(range(1, dataset_size + 1))

@app.route("/Search_algo", methods=["GET", "POST"])
def search_algo():
    test_data = ""
    array = []

    if request.method == "POST":
        try:
            array_str = request.form.get("array")
            target_str = request.form.get("target")
            search_type = request.form.get("search_type")

            dataset_size = int(request.form.get("dataset_size"))
            array = generate_array(dataset_size)
            test_data = ", ".join(map(str, array))

            target = int(target_str)
            low, high = 0, len(array) - 1

            result = -1  

            if search_type == "exponential":
                execution_time = timeit.timeit("exponential_search_wrapper(exponential_search, array, target)",
                                               globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = exponential_search_wrapper(binary_search, array, target)
            elif search_type == "binary":
                execution_time = timeit.timeit("binary_search_wrapper(binary_search, array, target)",
                                               globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = binary_search_wrapper(binary_search, array, target)
            elif search_type == "interpolation":
                execution_time = timeit.timeit("interpolation_search_wrapper(interpolation_search, array, target)",
                                               globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = interpolation_search_wrapper(interpolation_search, array, target)
            elif search_type == "jump":
                execution_time = timeit.timeit("jump_search_wrapper(jump_search, array, target)",
                                               globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = jump_search_wrapper(jump_search, array, target)
            elif search_type == "linear":
                execution_time = timeit.timeit("linear_search_wrapper(linear_search, array, target)",
                                               globals={**globals(), "array": array, "target": target}, number=1) * 1000
                result = linear_search_wrapper(linear_search, array, target)
            elif search_type == "ternary":
                execution_time = timeit.timeit("ternary_search_wrapper(ternary_search, array, target, low, high)",
                                               globals={**globals(), "array": array, "target": target, "low": low,
                                                        "high": high}, number=1) * 1000
                result = ternary_search_wrapper(ternary_search, array, target, low, high)

            return render_template("Search_algo.html", result=result, search_type=search_type, execution_time=execution_time,
                                   test_data=test_data)
        except ValueError:
            print(traceback.format_exc())  # Log the error for debugging
            return render_template("Search_algo.html", error="Invalid input. Ensure the array and target are integers.")

    return render_template("Search_algo.html", test_data=test_data, array=array)

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()

    if not data or "array" not in data or "target" not in data:
        return jsonify({"error": "Invalid request data. Provide 'array' and 'target'."}), 400

    array = data["array"]
    target = data["target"]

    result_iterative = exponential_search(array, target)
    # result_recursive = exponential_search_recursive(array, target)

    return jsonify({
        "iterative_search_result": result_iterative,
        # "recursive_search_result": result_recursive
    })

if __name__ == "__main__":
    app.run(debug=True)