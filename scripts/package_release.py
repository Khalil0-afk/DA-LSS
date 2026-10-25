import argparse
import ast
import hashlib
import io
from pathlib import Path
import re
import tokenize
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    files = [root / name for name in ("README.md", "LICENSE.md", "pyproject.toml",
                                      ".gitignore", "requirements-tested.txt")]
    for directory in ("da_lss", "configs", "tests", "docs", "scripts", "examples", "supplementary", ".github"):
        files.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    files.extend(p for p in (root / "data").rglob("*.md") if p.is_file())
    for directory in ("MIT data", "HUST data"):
        files.extend(p for p in (root / "data" / directory).rglob("*.csv") if p.is_file())
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if path.suffix.lower() == ".pdf" and path.relative_to(root).as_posix() != "supplementary/Supplementary_Methods.pdf":
                raise ValueError("Only the generated supplementary PDF belongs in the release")
            if path.suffix.lower() in (".doc", ".docx", ".rtf", ".html", ".tex"):
                raise ValueError("Release prose must use Markdown")
            if path.suffix.lower() == ".txt" and path.name != "requirements-tested.txt":
                raise ValueError("Explanatory text and archived logs must use Markdown")
            if path.suffix in (".py", ".md", ".json", ".toml", ".yml", ".yaml", ".txt", ".csv"):
                source = path.read_text(encoding="utf-8-sig")
                if re.search(r"[\u3400-\u9fff]", source):
                    raise ValueError(f"Non-English CJK text in release file: {path.relative_to(root)}")
                if path.suffix == ".py":
                    tree = ast.parse(source)
                    if any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(source).readline)):
                        raise ValueError(f"Code comment in {path.relative_to(root)}")
                    if any(ast.get_docstring(n) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))):
                        raise ValueError(f"Code docstring in {path.relative_to(root)}")
            archive.write(path, "DA-LSS-v0.3-English-Times/" + path.relative_to(root).as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive integrity check failed")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(f"{len(files)} files; {output.stat().st_size} bytes; SHA-256 {digest}")


if __name__ == "__main__":
    main()
