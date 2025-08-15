# Olive json Requirements

We have the following requirements for the olive json. They are also checked in `scripts\sanitize.py`:

- engine should not be used: place everything in the root to easy update for history generation
- separate quantization and evaluation datasets: so user could control them separately
- only one system is used: to allow select from different target runtime
