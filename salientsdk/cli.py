#!/usr/bin/env python
# Copyright Salient Predictions 2025

"""Command line interface for the Salient SDK.

Usage:
```
salientsdk examples
salientsdk version
```

"""

from salientsdk.__main__ import main as main_logic


def main():
    """Dispatch to __main__.py."""
    main_logic()


if __name__ == "__main__":
    main()
