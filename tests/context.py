"""Provide nlns package to tests without explicitly installing it."""

import os
import sys
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import nlns             # NOQA
