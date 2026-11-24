"""General entry point: python -m true_experience.experience VIDEO [options].

Same behaviour as ``true_experience.local_trailer`` (kept for compatibility).
"""
from .local_trailer import main

if __name__ == "__main__":
    raise SystemExit(main())
