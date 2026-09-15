"""Independent answer checks for the arithmetic calibration candidates."""
import ast
import calendar
import math
import operator
from datetime import date

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
       ast.LShift: operator.lshift, ast.RShift: operator.rshift}


def integer_expression(expression):
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError(f'Unsupported arithmetic syntax: {ast.dump(node)}')
    return visit(ast.parse(expression, mode='eval'))


def validate_item(task, item):
    """Raise on a bad oracle; return whether an independent check is implemented."""
    meta = item.get('metadata', {})
    answer = item['answer']
    if task in {'basic_arithmetic', 'chain_sum', 'products'}:
        expected = str(integer_expression(meta['expression']))
    elif task == 'lcm':
        expected = str(math.lcm(*meta['numbers']))
    elif task == 'count_bits':
        expected = str(int(meta['number']).bit_count())
    elif task == 'bitwise_arithmetic':
        expected = hex(integer_expression(meta['problem']))
    elif task == 'power_function':
        expected = str(math.pow(meta['base'], meta['exponent']))
    elif task == 'calendar_arithmetic' and meta['task'] == 'weekday_of_date':
        expected = date.fromisoformat(meta['target_date']).strftime('%A')
    elif task == 'calendar_arithmetic' and meta['task'] == 'weekday_of_date_from_first_date':
        target = date.fromisoformat(meta['target_date'])
        delta = (target - date(target.year, 1, 1)).days
        expected = calendar.day_name[(list(calendar.day_name).index(meta['first_day']) + delta) % 7]
    else:
        return False
    assert str(answer) == expected, (task, item['question'], answer, expected)
    return True
