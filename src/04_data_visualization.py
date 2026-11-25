"""Phase 3: generate all exploratory charts.

Run from the project root (after cleaning):
    python src/04_data_visualization.py

Charts are saved as 200-dpi PNG files in outputs/charts/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import charts
import utils


def main() -> None:
    utils.ensure_directories()
    df1, _ = utils.load_cleaned()
    print("Generating exploratory charts...")
    paths = []
    for chart_function in charts.EDA_CHARTS:
        print(f"  - {chart_function.__name__}")
        fig = chart_function(df1)
        filename = charts._expected_filename(chart_function.__name__)
        paths.append(utils.CHART_DIR / filename)
        utils.plt.close(fig)

    print(f"\nGenerated {len(paths)} charts:")
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Chart was not created: {path}")
        print(f"  {path.name} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001
        print(f"\nERROR: {error}", file=sys.stderr)
        raise
