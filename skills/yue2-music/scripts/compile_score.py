"""Compatibility entry point; canonical implementation is lyra.music_tools.compile_score."""
from lyra.music_tools.compile_score import *  # noqa: F403

if __name__ == "__main__":
    import sys
    from lyra.music_tools.compile_score import main
    sys.exit(main())
