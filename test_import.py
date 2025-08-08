import sys
import logging

logging.basicConfig(level=logging.DEBUG)

print("Python version:", sys.version)
print("sys.path:", sys.path)

try:
    from playwright_stealth import stealth_sync
    print("Import lykkedes! stealth_sync er tilgængelig.")
except ImportError as e:
    print("Import mislykkedes med fejl:", e)
    # Tilføj ekstra debug: Tjek om modulet findes manuelt
    try:
        import playwright_stealth
        print("Modulet 'playwright_stealth' findes, men stealth_sync mangler muligvis.")
    except ImportError as inner_e:
        print("Modulet 'playwright_stealth' findes slet ikke:", inner_e)