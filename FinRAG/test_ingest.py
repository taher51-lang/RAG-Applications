import sys
import os
from main import NyayaSetu
try:
    ns = NyayaSetu()
    ns.ingest("Bangalore_water_supply_1978.PDF", "landmark", "Banglore_water_supply_1978")
    print("Ingestion successful.")
except Exception as e:
    import traceback
    traceback.print_exc()
