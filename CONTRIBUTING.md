# Contributing

Thanks for helping with frads-gym. Bug reports, questions and pull requests are welcome on
[GitHub](https://github.com/silaskalmbach/frads-gym/issues).

## Reporting a problem

Please include the frads-gym, frads and Python versions, your operating system, the full error message
and, if possible, a configuration that reproduces it. `python -c "import frads, frads_gym; print(frads.__version__)"`
prints the frads version; the fork reports a different set of features (see the documentation).

## Development setup

```bash
git clone https://github.com/silaskalmbach/frads-gym.git
cd frads-gym
pip install -e ".[test]"
python -m pytest -m "not slow" tests
```

`-m slow` runs the tests that start EnergyPlus and Radiance (a few minutes). To test against the frads
fork, add `pip install -r requirements-5pm.txt`.

The documentation is built with MkDocs:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Pull requests

- Keep a pull request to one change and describe why it is needed.
- Add or adapt a test that fails without the change.
- Behaviour that changes simulation results must be switched on explicitly (a new option with the old
  behaviour as default), so that existing studies stay reproducible. Say so in the pull request.
- Update `CHANGELOG.md` under "Unreleased" and the documentation if a user-facing behaviour changes.
- CI runs the unit tests on Linux, Windows and macOS with Python 3.10 to 3.12 and both frads variants.

## Code of conduct

This project follows the [Contributor Covenant](https://github.com/silaskalmbach/frads-gym/blob/main/CODE_OF_CONDUCT.md).
