# build_snapshot.py, instructor utility: pre-warm the soil snapshot
#
# Purpose: populate soil_snapshot.json with the course reference points so
# that the snapshot fallback in soil_tool.py works out of the box, even on
# a machine that has never made a successful live call (weak WiFi, expired
# token, API outage during a live demo).
#
# Run this ONCE with a live connection and valid iSDA credentials in .env,
# then commit the resulting soil_snapshot.json to the repo. The file
# contains only public CC-BY 4.0 soil data, no credentials.
#
#   python build_snapshot.py

from soil_tool import get_soil

# Course reference points (see lab script for the pedagogical roles):
#   (-1.94, 30.06)   Kigali area. The recurring anchor point of the course.
#                    Expected topsoil pH: 6.7 (68% CI 6.4 to 7.0).
#   (-1.776, 29.728) Rural Rwanda point from the instructor smoke test.
#                    Expected topsoil pH: about 5.1. The contrast to Kigali
#                    is the "a point value is not a regional claim" lesson
#                    (30 m resolution cuts both ways).
POINTS = [(-1.94, 30.06), (-1.776, 29.728)]

# Properties the course docstrings advertise (soil() in tools.py):
PROPS = ["ph", "carbon_organic", "nitrogen_total"]

if __name__ == "__main__":
    ok, failed = 0, 0
    for lat, lon in POINTS:
        for prop in PROPS:
            try:
                result = get_soil(lat, lon, prop)
                tail = result[-60:].replace("\n", " ")
                print(f"OK   ({lat}, {lon}) {prop:15s} ...{tail}")
                ok += 1
            except Exception as exc:
                print(f"FAIL ({lat}, {lon}) {prop:15s} {type(exc).__name__}: {exc}")
                failed += 1
    print(f"\n{ok} entries cached, {failed} failed. "
          f"Snapshot file: soil_snapshot.json")
    if failed == 0:
        print("Next step: git add soil_snapshot.json && git commit")
