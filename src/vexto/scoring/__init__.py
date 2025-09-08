import logging, sys

# Sikr UTF-8 i konsollen (Windows/PowerShell venligt)
for _s in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
    try:
        if _s and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Tving egne handlers med eksplicit encoding
root = logging.getLogger()
root.handlers.clear()

stream = logging.StreamHandler(stream=sys.stdout)
stream.setLevel(logging.INFO)
stream.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

root.setLevel(logging.INFO)
root.addHandler(stream)