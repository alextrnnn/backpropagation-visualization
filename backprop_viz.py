import re
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

def simplify_expression(expr, forward_values):
    # Remove spaces from the input
    expr = expr.replace(' ', '')
    
    # Define patterns for each level of precedence in the correct order
    operators = [
        (r'(\w+\s*\^\s*\w+)', '^'),     # Power 
        (r'(\w+\s*[\*/]\s*\w+)', '*/'), # Multiplication and Division
        (r'(\w+\s*[\+-]\s*\w+)', '+-')  # Addition and Subtraction
    ]
    
    intermediate_steps = []
    intermediate_var = 'v'
    index = 1
    
    nodes = []
    links = []
    backward_gradients = {}

    # Helper function to compute an operation
    def compute_operation(op, left, right):
        if op == '+':
            return left + right
        elif op == '-':
            return left - right
        elif op == '*':
            return left * right
        elif op == '/':
            return left / right
        elif op == '^':
            return left ** right
        else:
            raise ValueError(f'Unsupported operator: {op}')

    for var, value in forward_values.items():
        nodes.append({
            'id': var,
            'label': var,
            'forward_value': value,
            'backward_gradient': 0.0,  
        })

    # Helper function to process a simple expression without parentheses
    def process_simple_expression(simple_expr):
        nonlocal index
        for pattern, ops in operators:
            while re.search(pattern, simple_expr):
                match = re.search(pattern, simple_expr)
                sub_expr = match.group(0)
                var_name = f'{intermediate_var}{index}'
                
                # Parse operation
                for op in ops:
                    if op in sub_expr:
                        operands = sub_expr.split(op)
                        left, right = operands[0].strip(), operands[1].strip()
                        operation = op
                        break
                
                # Compute forward pass value
                forward_values[var_name] = compute_operation(operation, forward_values[left], forward_values[right])
                
                # Avoid duplicate variable by only creating a new one if necessary
                if sub_expr not in [step.split(' = ')[1] for step in intermediate_steps]:
                    intermediate_steps.append(f'{var_name} = {sub_expr}')
                    
                    # Add the node for the new variable
                    nodes.append({
                        'id': var_name, 
                        'label': sub_expr, 
                        'forward_value': forward_values[var_name],
                        'backward_gradient': 0.0, 
                    })
                    
                    # Add links from operands to the new variable
                    for operand in [left, right]:
                        links.append({'source': operand, 'target': var_name, 'operation': operation})
                    
                    # Replace the expression with the variable
                    simple_expr = simple_expr[:match.start()] + var_name + simple_expr[match.end():]
                    index += 1
                else:
                    # Replace with the existing variable name
                    existing_var = [step.split(' = ')[0] for step in intermediate_steps if step.split(' = ')[1] == sub_expr][0]
                    simple_expr = simple_expr[:match.start()] + existing_var + simple_expr[match.end():]
                    
        return simple_expr

    while '(' in expr:
        # Find innermost parenthesis expression
        inner_expr = re.search(r'\([^()]+\)', expr).group(0)
        # Remove parentheses
        inner_expr_content = inner_expr[1:-1]
        # Process the content inside the parentheses
        result = process_simple_expression(inner_expr_content)
        # Replace the parentheses expression in the main expression with its result
        expr = expr.replace(inner_expr, result, 1)
    
    # Process remaining expression outside parentheses
    expr = process_simple_expression(expr)

    # Compute backward pass
    backward_gradients[expr] = 1.0  
    for step in reversed(intermediate_steps):
        var_name, sub_expr = step.split(' = ')
        for op in '+-*/^':
            if op in sub_expr:
                left, right = sub_expr.split(op)
                left, right = left.strip(), right.strip()
                
                if left not in backward_gradients:
                    backward_gradients[left] = 0.0
                
                if right not in backward_gradients:
                    backward_gradients[right] = 0.0
                
                # Propagate gradients based on operation
                if op == '+':
                    backward_gradients[left] += 1
                    backward_gradients[right] += 1
                elif op == '-':
                    backward_gradients[left] += 1
                    backward_gradients[right] += - 1
                elif op == '*':
                    backward_gradients[left] += forward_values[right]
                    backward_gradients[right] += forward_values[left]
                elif op == '/':
                    backward_gradients[left] += 1.0 / forward_values[right]
                    backward_gradients[right] += -forward_values[left] / (forward_values[right] ** 2)
                elif op == '^':
                    backward_gradients[left] += forward_values[right] * (forward_values[left] ** (forward_values[right] - 1))
                    backward_gradients[right] += forward_values[left] ** forward_values[right] * np.log(forward_values[left])
                    
                backward_gradients[left] *= backward_gradients[var_name]
                backward_gradients[right] *= backward_gradients[var_name]

    # Update gradients in nodes
    for node in nodes:
        node['backward_gradient'] = backward_gradients[node['id']]
    
    # Create a dictionary to store incoming links for each node
    incoming_links = {node['id']: [] for node in nodes}
    for link in links:
        incoming_links[link['target']].append(link['source'])

    # Use BFS to assign levels, starting from the result (final node). This is to make the graph easier to read.
    result_node = nodes[-1]['id']
    node_levels = {node['id']: -1 for node in nodes} 

    # Set the level of the result node to 0
    node_levels[result_node] = 0
    queue = deque([result_node])

    # BFS to define levels
    while queue:
        current_node = queue.popleft()
        current_level = node_levels[current_node]
        
        for dependency in incoming_links[current_node]:
            if node_levels[dependency] == -1:
                node_levels[dependency] = current_level + 1
                queue.append(dependency)

    for node in nodes:
        node['level'] = node_levels[node['id']]

    graph = {'nodes': nodes, 'links': links}
    return graph

expression = input('Enter an expression: ')

# Extract variables from the expression
variables = set(re.findall(r'\b[a-zA-Z_]\w*\b', expression))

# Prompt the user for variable values
forward_values = {}
for var in variables:
    forward_values[var] = float(input(f'Enter value for {var}: '))

# Simplify and process the expression
graph_data = simplify_expression(expression, forward_values)

def draw_computational_graph(graph_data, expr):
    G = nx.DiGraph()

    for node in graph_data['nodes']:
        G.add_node(node['id'], label=node['label'], value=node['forward_value'], level=node['level'])

    for link in graph_data['links']:
        G.add_edge(link['source'], link['target'], operation=link['operation'])

    # Set the position of nodes using multipartite layout (grouping by level)
    pos = nx.multipartite_layout(G, subset_key='level')

    plt.figure(figsize=(12, 12))
    node_colors = ['yellow' if node['id'] == graph_data['nodes'][-1]['id'] else 'skyblue' for node in graph_data['nodes']]
    nx.draw(G, pos, with_labels=True, node_size=2500, node_color=node_colors, font_size=10, font_weight='bold', arrows=True)

    # Annotate edges with the forward values and backward gradient
    for node in graph_data['nodes']:
        node_pos = pos[node['id']]

        isFinal = '' if node['id'] != graph_data['nodes'][-1]['id'] else 'final_val = '
        plt.text(node_pos[0], node_pos[1] + 0.05, f'{isFinal}{node['label']} = {node['forward_value']}', fontsize=12, ha='center', color='green')
        plt.text(node_pos[0], node_pos[1] - 0.1, f'Backward Grad: {node['backward_gradient']}', fontsize=12, ha='center', color='red')

    plt.title(f'Computational graph for {expr}')
    plt.show()

draw_computational_graph(graph_data, expression)
