from pathlib import Path
import py_compile


def test_main_script_compiles():
    script = Path(__file__).resolve().parents[1] / "scripts" / "09_reproduce_all.py"
    py_compile.compile(str(script), doraise=True)
