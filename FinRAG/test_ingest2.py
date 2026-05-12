print("Starting import test...")
try:
    from main import NyayaSetu
    print("Import successful")
except Exception as e:
    import traceback
    print(f"Import failed: {e}")
    traceback.print_exc()