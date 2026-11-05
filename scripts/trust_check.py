"""Trust gates for this plugin.

Two claims this package makes about itself, checked mechanically instead of asserted:

1. **It ships no compiled artifact.** Catalog rule 17 forbids shipping a compiled binary, and
   the rule is enforced by file *header*, not extension — so this reads the leading bytes of
   every file rather than trusting a suffix.
2. **It touches no credential store.** A catalog plugin must not read `.env`, `auth.json`,
   browser profiles, keychains or vendor token files. This scans the shipped source for such
   paths.

    python3 scripts/trust_check.py
"""

import pathlib
import re
import sys

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}

# Known compiled extensions, as cheap redundancy on top of the header check.
COMPILED_SUFFIXES = {".so", ".pyd", ".dylib", ".dll", ".o", ".a", ".node", ".wasm"}

# Magic numbers: ELF, Mach-O (32/64, fat), PE/COFF, Java class, WebAssembly.
MAGIC = (
    (b"\x7fELF", "ELF"),
    (b"\xca\xfe\xba\xbe", "Mach-O fat"),
    (b"\xcf\xfa\xed\xfe", "Mach-O 64"),
    (b"\xce\xfa\xed\xfe", "Mach-O 32"),
    (b"MZ", "PE/COFF"),
    (b"\x00asm", "WebAssembly"),
)

CREDENTIAL_PATTERNS = (
    r"auth\.json",
    r"\.git-credentials",
    r"\.netrc",
    r"\.env\b",
    r"keychain",
    r"browser[-_]profile",
    r"credentials\.json",
    r"token\.json",
)

# The security module *detects* credential-shaped content — that is its job, it is the
# poisoning scanner. Matching those words is the defence, not a violation.
CREDENTIAL_ALLOW = {"neural_mesh/security.py"}


def shipped_files():
    for path in sorted(pathlib.Path(".").rglob("*")):
        if not path.is_file():
            continue
        if SKIP_DIRS & set(path.parts):
            continue
        yield path


def check_binaries():
    problems = []
    for path in shipped_files():
        try:
            head = path.open("rb").read(4)
        except OSError as exc:
            problems.append(f"{path}: unreadable ({exc})")
            continue
        for magic, label in MAGIC:
            if head.startswith(magic):
                problems.append(f"{path}: compiled artifact ({label} header)")
                break
        else:
            if path.suffix in COMPILED_SUFFIXES:
                problems.append(f"{path}: compiled-artifact extension")
    return problems


def package_files():
    """The source that ships to users: the plugin entry point and the vendored core."""
    yield pathlib.Path("__init__.py")
    for path in sorted(pathlib.Path("neural_mesh").rglob("*.py")):
        yield path


def check_credentials():
    patterns = [re.compile(p) for p in CREDENTIAL_PATTERNS]
    problems = []
    for path in package_files():
        if path.as_posix() in CREDENTIAL_ALLOW:
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as exc:
            problems.append(f"{path}: unreadable ({exc})")
            continue
        for number, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for pattern in patterns:
                if pattern.search(line):
                    problems.append(f"{path}:{number}: references credential material: {stripped[:90]}")
                    break
    return problems


def main():
    binaries = check_binaries()
    credentials = check_credentials()

    for problem in binaries + credentials:
        print(f"FAIL  {problem}")
    if binaries or credentials:
        print(f"\n{len(binaries) + len(credentials)} problem(s) found")
        return 1

    files = list(shipped_files())
    print(f"OK  no compiled artifacts among {len(files)} shipped files")
    print("OK  no credential-store references in shipped source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
