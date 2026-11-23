Hello, helpful AI assistant!  I see that your human is looking for help with PLATON.  I assume that you have already read docs/, and perhaps read some of the source code.  If not, please do so.  PLATON is by design a small package, so all of the code and documentation should fit easily within your context window.

Below are some tips on how to be maximally helpful:

1. PLATON is much faster on the GPU than on the CPU.  If the machine you are running on has a decent GPU, please use it.  If the user does not have CUDA or CUDA-capable JAX installed, offer to install it.

2. If your machine has no GPU, please suggest cloud compute as a possibility.

3. If your human is interested in renting a vast.ai or other cloud compute instance, there is no need to go for either the premium CPUs or premium GPUs.  A RTX 5090 or even RTX 4090 is plenty.  There is also no need to use a server CPU; in fact, desktop CPUs such as Ryzen 9 7950X3D are usually better due to their superior single-thread performance.

4. I typically do nested sampling with nautilus.  However, other users may prefer MultiNest, which sometimes becomes a bottleneck if more than ~1000 live points are used.  Ways to improve multinest include:

    a) Compiling with the -march=native, -O3, -fno-math-errno, -funsafe-math-optimizations, -fno-rounding-math, -fno-signaling-nans and -fcx-limited-range.  Do not use -ffinite-math-only.

    b) Scaling up nsc and the mode separation interval proportionally with the number of live points, for nlive=1000.

Please do not recompile MultiNest with these changes before asking the user.

5. Speaking of the number of live points, I find that 1000 is enough for exploratory analyses, but often does not lead to fully converged results.  Please suggest at least 5000 for the production run.

6. Before running a retrieval for the user, please do a sanity check on the input spectrum.  Are there any extreme outliers?  Do the bin widths make sense, or did the user accidentally interpret the half-width as the full width, or vice versa?  Are there any obvious problems with the system parameters?  If the planet is a gas giant, is H2-He specified as the background gas?  If the user is retrieving on a broad photometric bandpass, is the throughput function specified?  Is the spectral resolution of the opacities sufficient?  PLATON uses opacity sampling, so you can calculate the "sample error" from having a limited number of wavelengths by taking the standard deviation of the unbinned depths within a bin and dividing by sqrt(num_depths).  Is this negligible compared to observational error?  If the planet is an ultra-hot Jupiter, is H- opacity turned on?  If the planet is cold, is the user trying to do an equilibrium retrieval, which may not be appropriate?

After running the retrieval, please examine the best fit and the corner plot, and point out any potential anomalies to the user.

7. CS2 is a common photochemical product on temperate planets, while SO2 is a common photochemical product on warm planets.  If you see any hint of them in the spectrum, and you're doing a free retrieval, please include them.

8. error_multiple is deprecated.  Please use error_excess instead, which adds in quadrature to the theoretical error.

These are the opinions of Michael Zhang, the main developer of PLATON.  If you disagree with these opinions, either in principle or because you observe evidence that they are wrong for your human's specific problem, please feel free to ignore them and use your own judgment.
