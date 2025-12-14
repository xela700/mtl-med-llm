import subprocess
import sys

# Clean previous build
subprocess.run(
    [sys.executable, "-m", "sphinx", "-M", "clean", ".", "_build"],
    check=True,
)

# Build HTML docs
subprocess.run(
    [sys.executable, "-m", "sphinx", "-b", "html", ".", "_build/html"],
    check=True,
)