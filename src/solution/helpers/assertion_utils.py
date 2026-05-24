import re


NEG_LITERAL = re.compile(r'^-\d+(?:\.\d+)?[LlFfDd]?$')


def strip_trailing_comment(line):
    # Remove trailing // or /* comment, but only when outside a string literal.
    in_str = escape = False
    for i, ch in enumerate(line):
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str and ch == '/' and i + 1 < len(line) and line[i + 1] in ('/', '*'):
            return line[:i].rstrip()
    return line


def count_brackets(s, open_ch, close_ch):
    # Count occurrences of open_ch and close_ch outside string literals.
    open_count = close_count = 0
    in_str = None  # None, '"', or "'"
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            continue
        if ch == open_ch:
            open_count += 1
        elif ch == close_ch:
            close_count += 1
    return open_count, close_count


def split_args(s):
    # Split assertion args by top-level ',', respecting parens depth and string literals.
    args, depth, start = [], 0, 0
    in_str = None
    escape = False
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ('"', "'"):
            in_str = ch
            continue
        if ch in '({':
            depth += 1
        elif ch in ')}':
            depth -= 1
        elif ch == ',' and depth == 0:
            args.append(s[start:i])
            start = i + 1
    args.append(s[start:])
    return args


def wrap_negative_literals(assertion):
    # Wrap bare negative number literals in parentheses: -2L -> (-2L).
    match = re.match(r'(assert\w+\()(.+)(\);)$', assertion.rstrip())
    if not match:
        return assertion

    prefix, inner, suffix = match.group(1), match.group(2), match.group(3)
    args = split_args(inner)
    changed = False
    for i, arg in enumerate(args):
        stripped = arg.strip()
        if NEG_LITERAL.match(stripped):
            args[i] = f'({stripped})'
            changed = True
        else:
            args[i] = stripped
    if not changed:
        return assertion
    return prefix + ', '.join(args) + suffix


def fix_assertion(assertion, language='java'):
    if not assertion:
        return assertion

    assertion = assertion.strip()

    open_p, close_p = count_brackets(assertion, '(', ')')
    open_b, close_b = count_brackets(assertion, '{', '}')

    if '() -> {' in assertion and close_b < open_b:
        missing_b = open_b - close_b
        assertion = assertion.rstrip(';') + '}' * missing_b
        open_p, close_p = count_brackets(assertion, '(', ')')
        if close_p < open_p:
            assertion += ')' * (open_p - close_p)
    elif close_p < open_p:
        assertion = assertion.rstrip(';') + ')' * (open_p - close_p)

    if language.lower() != 'python':
        if not assertion.endswith(';'):
            assertion += ';'

    assertion = wrap_negative_literals(assertion)

    return assertion


def post_process_assertion(raw_assertion, language='java'):
    # Clean markdown/backtick noise, smart quotes; pick assertion line from multi-line;
    # strip trailing comment; balance parens/braces and wrap negative literals.
    if not raw_assertion:
        return ''

    raw_assertion = raw_assertion.strip()
    if not raw_assertion:
        return ''

    raw_assertion = re.sub(r'^```\w*\n?', '', raw_assertion)
    raw_assertion = raw_assertion.replace('`', '').strip()
    raw_assertion = raw_assertion.replace('“', '"').replace('”', '"')
    raw_assertion = raw_assertion.replace('‘', "'").replace('’', "'")

    for line in raw_assertion.split('\n'):
        line = line.strip()
        if line.startswith('assert') or line.startswith('expect('):
            raw_assertion = line
            break

    raw_assertion = strip_trailing_comment(raw_assertion)
    return fix_assertion(raw_assertion, language)
