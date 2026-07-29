"""Allow ``python -m rlattack`` to invoke the command-line interface."""

from rlattack.cli import main  # pragma: no cover

raise SystemExit(main())  # pragma: no cover
