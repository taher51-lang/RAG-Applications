import sys
import os
from main import NyayaSetu
try:
    ns = NyayaSetu()
    ns.ingest("Secretary_State_Of_Karnataka_And_vs_Umadevi.PDF", "citing", "Secretary_State_Of_Karnataka_And_vs_Umadevi")
    print("Ingestion successful.")
except Exception as e:
    import traceback
    traceback.print_exc()
