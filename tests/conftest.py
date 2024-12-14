import os
import sys
import warnings

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Filter out the specific deprecation warning from dateutil
warnings.filterwarnings(
    "ignore",
    message="datetime.datetime.utcfromtimestamp()",
    category=DeprecationWarning,
    module="dateutil.tz.tz",
)
