"""Restricted mathematical syntax, with SymPy notation transformations only.

No Python eval/exec: the transformed AST is interpreted with a small whitelist.
This is not a process sandbox or a guarantee of bounded symbolic computation.
"""
import ast
import keyword
import re

from sympy import Float, Integer, Rational, Symbol
from sympy.parsing.sympy_parser import stringify_expr


def parse_math(text, names, transformations):
    if len(text) > 2048:
        raise ValueError("Expression is too long (maximum 2048 characters).")
    if not re.fullmatch(r"[A-Za-z0-9_+*/().,\s-]+", text):
        raise ValueError("Use mathematical names, numbers, parentheses and arithmetic operators.")
    if re.search(r"(?<![0-9])\.(?![0-9])|[A-Za-z_)][.]|[.]\s*[A-Za-z_]", text):
        raise ValueError("Attribute access is not allowed.")
    if re.search(r',\s*\)', text):
        raise ValueError('Missing function argument.')
    depth = 0
    for ch in text:
        depth += (ch == '(') - (ch == ')')
        if depth < 0 or depth > 32:
            raise ValueError("Invalid or excessively nested parentheses.")
    if depth:
        raise ValueError("Unbalanced parentheses.")
    local = dict(names)
    for name in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text):
        if name in local:
            continue
        if name.startswith('_') or '__' in name or keyword.iskeyword(name) or len(name) > 32:
            raise ValueError("Invalid parameter name.")
        if name in {'eval', 'exec', 'open', 'getattr', 'globals', 'locals', 'Symbol',
                    'Integer', 'Float', 'Rational', 'Function', 'lambda', 'import'}:
            raise ValueError("Only mathematical functions may be called.")
        # a(t+1) remains implicit multiplication; unknown named function calls do not.
        if len(name) > 1 and re.search(r'\b' + re.escape(name) + r'\s*\(', text):
            raise ValueError("Unknown mathematical function.")
        local[name] = Symbol(name, real=True)
    constructors = {'Integer': Integer, 'Float': Float, 'Rational': Rational}
    source = stringify_expr(text, local, constructors, transformations)
    tree = ast.parse(source, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 512:
        raise ValueError("Expression is too complex.")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and node.id in local and not callable(local[node.id]):
            return local[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, str):
            # Strings only originate in SymPy's numeric transformations.
            if isinstance(node.value, str) and not re.fullmatch(r'[0-9.eE+-]+', node.value):
                raise ValueError("Invalid numeric literal.")
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.Pow):
                if getattr(right, 'is_number', False) and abs(complex(right)) > 1000:
                    raise ValueError("Numeric powers must have magnitude at most 1000.")
                return left ** right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            fn = constructors.get(node.func.id, local.get(node.func.id))
            if callable(fn) and 1 <= len(node.args) <= 2:
                return fn(*(visit(arg) for arg in node.args))
        raise ValueError("Unsupported mathematical syntax.")

    result = visit(tree)
    if not hasattr(result, 'free_symbols'):
        raise ValueError("Expected a mathematical expression.")
    return result
