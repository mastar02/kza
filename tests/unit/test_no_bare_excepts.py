import ast
import glob


def test_no_bare_excepts_in_src():
    """Verify no bare except: clauses exist in source code."""
    violations = []
    for filepath in glob.glob("src/**/*.py", recursive=True):
        with open(filepath) as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                violations.append(f"{filepath}:{node.lineno}")
    assert violations == [], f"Bare except: found at: {violations}"
