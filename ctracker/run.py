import sys
from ctracker import ctracker
if len(sys.argv) == 2:
    sys.path.append(str(sys.argv[1]))
    c = ctracker("configuration.py")
    c.run()
