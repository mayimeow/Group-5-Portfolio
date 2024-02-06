import random
import time
import timeit
import traceback
from flask import Flask, request, render_template, jsonify, redirect, url_for
from exponential_search import exponential_search, exponential_search_wrapper
from binary_search import binary_search, binary_search_wrapper
from interpolation_search import interpolation_search, interpolation_search_wrapper
from jump_search import jump_search, jump_search_wrapper
from linear_search import linear_search, linear_search_wrapper
from ternary_search import ternary_search, ternary_search_wrapper
from linked_list import Queue as MyQueue, Deque as MyDeque

app = Flask(__name__,)
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
mrt_graph = {
    'Roosevelt' : ['Balintawak'],
    'Balintawak' : ['Roosevelt', 'Malvar'],
    'Malvar' : ['Balintawak', 'Monumento'],
    'Monumento' : ['Malvar', '5th Avenue'],
    '5th Avenue' : ['Monumento', 'R.Papa'],
    'R.Papa' : ['5th Avenue', 'Abad Santos'],
    'Abad Santos' : ['R.Papa', 'Blumentritt'],
    'Blumentritt' : ['Abad Santos', 'Tayuman'],
    'Tayuman' : ['Blumentritt', 'Bambang'],
    'Bambang' : ['Tayuman', 'D-Jose/Recto'],
    'D-Jose/Recto' : ['Bambang', 'Carriedo', 'Legarda'],
    'Carriedo' : ['D-Jose/Recto', 'Central Terminal'],
    'Central Terminal' : ['Carriedo', 'United Nations'],
    'United Nations' : ['Central Terminal', 'Pedro Gil'],
    'Pedro Gil' : ['United Nations', 'Quirino Avenue'],
    'Quirino Avenue' : ['Pedro Gil', 'Vito Cruz'],
    'Vito Cruz' : ['Quirino Avenue', 'Gil Puyat'],
    'Gil Puyat' : ['Vito Cruz', 'Libertad'],
    'Libertad' : ['Gil Puyat', 'EDSA/Taft'],
    'EDSA/Taft' : ['Libertad', 'Baclaran', 'Magallanes'],
    'Baclaran' : ['EDSA/Taft'],
    'Santolan' : ['Katipunan'],
    'Katipunan' : ['Santolan', 'Anonas'],
    'Anonas' : ['Katipunan', 'Araneta Center-Cubao'],
    'Araneta Center-Cubao': ['Anonas', 'Betty Go-Belmonte', 'GMA Kamuning', 'Santolan-Anapolis'],
    'Betty Go-Belmonte' : ['Araneta Center-Cubao', 'Gilmore'],
    'Gilmore' : ['Betty Go-Belmonte', 'J.Ruiz'],
    'J.Ruiz' : ['Gilmore', 'V.Mapa'],
    'V.Mapa' : ['J.Ruiz', 'Pureza'],
    'Pureza': ['V.Mapa', 'Legarda'],
    'Legarda': ['Pureza', 'D-Jose/Recto'],
    'Quezon Avenue': ['GMA Kamuning'],
    'GMA Kamuning': ['Quezon Avenue', 'Araneta Center-Cubao'],
    'Santolan-Anapolis': ['Araneta Center-Cubao', 'Ortigas'],
    'Ortigas': ['Santolan-Anapolis', 'Shaw Boulevard'],
    'Shaw Boulevard': ['Ortigas', 'Boni'],
    'Boni': ['Shaw Boulevard', 'Guadalupe'],
    'Guadalupe': ['Boni', 'Buendia'],
    'Buendia': ['Guadalupe', 'Ayala'],
    'Ayala': ['Buendia', 'Magallanes'],
    'Magallanes': ['Ayala', 'EDSA/Taft'],
}

def dijkstra(graph, start, end):
    queue = [(0, start, [])]
    visited = set()

    while queue:
        (cost, node, path) = queue.pop(0)

        if node not in visited:
            visited.add(node)
            path = path + [node]

            if node == end:
                return path

            for next_node in graph[node]:
                if next_node not in visited:
                    queue.append((cost + 1, next_node, path))

    return None

@app.route('/Graph')
def Graph():
    return render_template('Graph.html')

@app.route('/find_shortest_path', methods=['POST'])
def find_shortest_path():
    source = request.form['source']
    destination = request.form['destination']

    shortest_path = dijkstra(mrt_graph, source, destination)

    if shortest_path:
        result = f'Shortest path: {" -> ".join(shortest_path)}'
    else:
        result = 'No path found.'

    return jsonify({'result': result})
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i+1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
    return arr

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key
    return arr

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

    return arr

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def get_user_input(size_option):
    sizes = {1: 10, 2: 100, 3: 1000, 4: 10000}

    if size_option in sizes:
        size = sizes[size_option]
        array = [random.randint(1, 1000) for _ in range(size)]
        return array
    else:
        return []

def display_array(arr):
    print("\nRandomly Generated Array:")
    print(arr)

@app.route('/sort_page', methods=['GET', 'POST'])
def sort_page():
    if request.method == 'POST':
        array_size = int(request.form['size'])
        array = [random.randint(1, 1000) for _ in range(array_size)]
        sorting_algorithm = request.form['algorithm']

        sorting_algorithms = {
           'bubble_sort': bubble_sort,
           'selection_sort': selection_sort,
           'insertion_sort': insertion_sort,
           'merge_sort': merge_sort,
           'quick_sort': quick_sort,
       }

        if sorting_algorithm not in sorting_algorithms:
            return render_template('sort.html', error="Invalid sorting algorithm selected.")

        start_time = time.time()

        if sorting_algorithm in sorting_algorithms:
            sorted_array = sorting_algorithms[sorting_algorithm](array.copy())
        else:
            return render_template('sort.html', error="Invalid sorting algorithm selected.")

        end_time = time.time()

        return render_template('sort.html', array=array, sorted_array=sorted_array, time_taken=end_time - start_time)

    return render_template('sort.html')

# Hash table initialization
hash_table = [[] for _ in range(32)]

# Hash function
def hash_function_1(key):
    return key % 32

def hash_function_2(key):
    return ((1731 * key + 520123) % 524287) % 32
def hash_function_3(key):
    return hash(key) % 32

@app.route('/hash')
def hash():
    return render_template('hash.html')




@app.route('/process', methods=['POST'])
def process():
    global hash_table  


    selected_hash_function = request.form.get('hash_function')
    num_commands = int(request.form.get('num_commands'))
    commands = request.form.get('commands').split('\n')


    
    hash_table = [[] for _ in range(32)]


    
    def insert_into_hash_table(word):
        word = word.strip()  
        key = sum(ord(char) for char in word)
        if selected_hash_function == 'Hash Function 1':
            index = hash_function_1(key)
        elif selected_hash_function == 'Hash Function 2':
            index = hash_function_2(key)
        else:
            index = hash_function_3(key)


        hash_table[index].insert(0, word)


   
    def delete_from_hash_table(word):
        word = word.strip()  
        key = sum(ord(char) for char in word)
        if selected_hash_function == 'Hash Function 1':
            index = hash_function_1(key)
        elif selected_hash_function == 'Hash Function 2':
            index = hash_function_2(key)
        else:
            index = hash_function_3(key)


        if word in hash_table[index]:
            hash_table[index].remove(word)



    for command in commands:
        if command.startswith('del '):
            delete_from_hash_table(command[4:])
        else:
            insert_into_hash_table(command)
    
    return render_template('hash.html', hash_table=hash_table)

@app.route('/output')
def output():
    return render_template('output.html', hash_table=hash_table)
if __name__ == "__main__":
    app.run(debug=True)