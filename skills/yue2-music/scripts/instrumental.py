"""Compatibility entry point; canonical implementation is lyra.music_tools.instrumental."""
from lyra.music_tools.instrumental import *  # noqa: F403

if __name__ == "__main__":
    import sys
    from lyra.music_tools.instrumental import main
    sys.exit(main())
