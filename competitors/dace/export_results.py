"""Copy all three workloads' enumerate results into the main experiments directory."""

import argparse
from pathlib import Path
import shutil


DACE_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = DACE_ROOT.parents[1] / "experiments" / "competitors_results" / "dace"
QUERY_SETS = ("job_light", "stats_ceb", "ergastf1")
RESULT_TYPES = ("plan_hints", "execution_results")


def export_results(workloads_dir: Path, output_dir: Path) -> None:
    copies = []
    for query_set in QUERY_SETS:
        for result_type in RESULT_TYPES:
            filename = f"{query_set}_{result_type}.csv"
            source = workloads_dir / query_set / "enumerate" / filename
            destination = output_dir / query_set / filename
            copies.append((source, destination))

    missing = [str(source) for source, _ in copies if not source.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run planning and execution for all three workloads before exporting. "
            "Missing result files:\n" + "\n".join(missing)
        )

    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"Exported {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        export_results(DACE_ROOT / "workloads", RESULTS_ROOT)
    except FileNotFoundError as error:
        parser.exit(status=1, message=f"{error}\n")


if __name__ == "__main__":
    main()
