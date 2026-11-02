from pathlib import Path
from bse import BSE

folder = Path(".cache") / "bse_api_test"
folder.mkdir(parents=True, exist_ok=True)

with BSE(download_folder=str(folder)) as api:
    for code in ("500179", "532540"):
        print(f"\nTEST BSE scrip code: {code}")
        try:
            result = api.equityPriceVolumeT12M(code)
            print("Result type:", type(result).__name__)
            print("Result:", str(result)[:1000])
        except Exception as exc:
            print("ERROR:", type(exc).__name__, str(exc)[:300])
