See [the reproduction report](../../docs/benchmark-reproduction.md) for contracts,
source selection, commands, actual results, and scientific limitations.

Python 3 standard library only. Build the Rust example first. C# is an optional
external prerequisite, configured with `SECA_REFERENCE_ROOT`. The runner never
builds, modifies, or copies the C# reference. Missing reference execution is a
reported failure; committed goldens are never substituted for actual execution.
