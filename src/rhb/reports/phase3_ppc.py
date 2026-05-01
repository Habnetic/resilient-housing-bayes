from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt


CITIES = ["RTM", "HAM", "DON"]

BASE_PATH = Path("outputs") / "phase3"


def plot_ppc_for_city(city: str) -> None:
    city = city.upper()

    idata_path = BASE_PATH / city / "idata.nc"
    out_path = BASE_PATH / city / f"ppc_{city}.png"

    if not idata_path.exists():
        raise FileNotFoundError(f"Missing idata for {city}: {idata_path}")

    print(f"[ppc] loading {idata_path}")
    idata = az.from_netcdf(idata_path)

    print(f"[ppc] plotting {city}")
    az.plot_ppc(idata)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[ppc] saved → {out_path}")


def main():
    print("[ppc] starting PPC plots")

    for city in CITIES:
        print(f"\n[city] {city}")
        plot_ppc_for_city(city)

    print("\n[ppc] done")


if __name__ == "__main__":
    main()