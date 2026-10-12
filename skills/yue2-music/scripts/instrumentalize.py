"""Compatibility entry point; canonical implementation is lyra.music_tools.instrumentalize."""
from lyra.music_tools.instrumentalize import *  # noqa: F403

if __name__ == "__main__":
    import sys
    from lyra.music_tools.instrumentalize import main
    sys.exit(main())
