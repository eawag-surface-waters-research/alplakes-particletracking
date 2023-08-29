import os
import subprocess
from datetime import datetime
from preprocess import preprocess
from functions import random_points_in_circle, plot_particle_tracking

lakes = [
    {"name": "ageri", "particles": [689500, 220000], "type": "curvilinear"},
    {"name": "biel", "particles": [580269, 215464], "type": "curvilinear"},
    {"name": "caldonazzo", "particles": [674000, 5098000], "type": "cartesian"},
    {"name": "garda", "particles": [628000, 5050000], "type": "curvilinear"},
    {"name": "geneva", "particles": [531350, 145124], "type": "curvilinear"},
    {"name": "greifensee", "particles": [693199, 245870], "type": "curvilinear"},
    {"name": "hallwil", "particles": [658735, 237277], "type": "cartesian"},
    {"name": "joux", "particles": [511380, 165584], "type": "curvilinear"},
    {"name": "lugano", "particles": [718659, 94426], "type": "curvilinear"},
    {"name": "murten", "particles": [572990, 198024], "type": "curvilinear"},
    {"name": "stmoritz", "particles": [784814, 152079], "type": "cartesian"},
    {"name": "zurich", "particles": [697899, 230725], "type": "curvilinear"}
]

start = datetime(2023, 7, 2, 6)
end = datetime(2023, 7, 3, 6)

print("Running particle tracking tests for {} lakes".format(len(lakes)))

failed = []
for lake in lakes:
    try:
        print("{}: Creating inputs".format(lake["name"]))
        particles = random_points_in_circle(lake["particles"][0], lake["particles"][1], 100, 10000, 0.5, 10)
        out = preprocess("setup_verification", lake["name"], start, end, particles, grid_type=lake["type"])
        command = "conda run -n ctracker python run.py {}".format(out["working_dir"])
        script_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ctracker")
        print("{}: Running particle tracking".format(lake["name"]))
        subprocess.run(command, check=True, shell=True, cwd=script_directory)
        print("{}: Plotting outputs".format(lake["name"]))
        plot_particle_tracking(out["working_dir"], out["x0"], out["y0"], out["x1"], out["y1"], lake["name"], save=True, plot=False, grid_type=lake["type"])
        print("{}: Complete".format(lake["name"]))
    except Exception as e:
        failed.append(lake["name"])
        print(e)
        print("{}: Failed.".format(lake["name"]))

if len(failed) > 0:
    raise ValueError("Tests failed for lakes: {}".format(", ".join(failed)))
else:
    print("All tests passed.")
