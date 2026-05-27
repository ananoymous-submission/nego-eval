"""Persistent storage directory management for HF Spaces."""
import os


def ensure_storage_dirs():
    """Create data directory structure.

    CRITICAL: Create parent directory FIRST before subdirectories.
    This must be called before DatabaseService initialization, or SQLite
    will fail to create the database file.
    """
    # Create parent directory FIRST (or DatabaseService will fail)
    os.makedirs("data", exist_ok=True)

    # Then create subdirectories
    os.makedirs("data/user_profiles", exist_ok=True)
    os.makedirs("data/session_logs", exist_ok=True)

    print("✓ data directories created")
