## MsLightweaver2d

Simple application of the [Lightweaver framework](https://github.com/Goobley/Lightweaver) and [MsLightweaver](https://github.com/Goobley/MsLightweaver) to investigate the radiative effects of a flaring atmosphere adjacent to a two-dimensional slab of quiet Sun. The slab is held static with a fixed temperature profile, but the electron density can be allowed to vary through charge conservation.
Results will be described in my thesis.

Note: you will need an `atmost.dat`  (`IATMT = 1` in `param.dat`) from RADYN, the CDF alone is insufficient, as every timestep is needed.