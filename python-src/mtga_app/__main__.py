import sys
from multiprocessing import freeze_support

from mtga_app import main  # type: ignore[reportMissingImports]

freeze_support()

if __name__ == "__main__":
    sys.exit(main())
