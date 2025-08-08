import importlib.util, traceback

spec = importlib.util.find_spec("sitecustomize")
print("find_spec ->", spec)

try:
    import sitecustomize
    print("IMPORT OK – path:", sitecustomize.__file__)
except Exception:
    print("IMPORT FAILED – traceback:")
    traceback.print_exc()
