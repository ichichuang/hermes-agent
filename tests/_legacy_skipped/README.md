Legacy-skipped hermes-agent tests live here intentionally.

These files were moved out of the active production suite because they are
legacy-baseline instability, not sanitizer or gateway-regression coverage.
They remain in-repo for later rehabilitation, but production validation does
not collect them.

Active production runs use `--strict-production` to allow only approved
environment/platform skips in the live test tree and to fail on new
unclassified skips.
