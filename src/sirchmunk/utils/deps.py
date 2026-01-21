import shutil
import warnings

def check_dependencies() -> bool:
    """
    Check if required dependencies 'rg' (ripgrep) and 'rga' (ripgrep-all) are installed.
    """

    if not shutil.which("rg"):
        warnings.warn(
            "\n\n"
            "⚠️  [Sirchmunk Warning] Missing dependency: 'rg' (ripgrep).\n"
        )
        return False

    if not shutil.which("rga"):
        warnings.warn(
            "\n\n"
            "⚠️  [Sirchmunk Warning] Missing dependency: 'rga' (ripgrep-all).\n"
        )
        return False

    return True
