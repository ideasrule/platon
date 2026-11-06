---
title: 'PLATON 7: An ultra-fast exoplanet atmospheric retrieval code'
tags:
  - Python
  - astronomy
  - exoplanets
  - exoplanet atmospheres
authors:
  - name: Michael Zhang
    orcid: 0000-0002-0659-1783
    equal-contrib: false
    corresponding: true
    affiliation: 1
    
  - name: Tyler Fairnington
    equal-contrib: false
    affiliation: 1
    
  - name: Yayaati Chachan
    affiliation: 2
    equal-contrib: false
    
  - name: Kimberly Paragas
    equal-contrib: false
    affiliation: 3
    
  - name: Eliza Kempton
    affiliation: 1
    equal-contrib: false
    
  - name: Heather Knutson
    affiliation: 3
    equal-contrib: false
    
  - name: Jacob Bean
    affiliation: 1
    equal-contrib: false


affiliations:
 - name: University of Chicago
   index: 1
   ror: 024mw5h28
 - name: UC Santa Cruz
   index: 2
 - name: Caltech
   index: 3
 

date: 6 November 2026
bibliography: paper.bib

---

# Summary
The atmospheres of hundreds of exoplanets have been observed with transmission or
emission spectroscopy.  To derive composition, temperature, cloudtop pressures, and other
properties from these observations, the most common tool is a retrieval: an
a parameterized atmospheric model, coupled with a Bayesian exploration of parameter space
using nested sampling or MCMC.  With the launch of JWST and proliferation of
high-resolution spectrographs on ground telescopes, the increasing precision and volume
of exoplanet atmospheric spectra require retrieval codes that are fast even at
high spectral resolution.

# Statement of need
PLATON is an open source atmospheric retrieval Python code first released in 2018 [@Zhang:2019] that
was greatly expanded in @Zhang:2020 and @Zhang:2025.  The forward model can generate transmission and emission
of both rocky and gaseous exoplanets.  The retrieval supports both equilibrium chemistry
and free retrieval using either nested sampling or MCMC.

PLATON 7 is a complete rewrite of PLATON 6 using Claude Fable 5.  The biggest changes
are that PLATON now uses JAX [@Bradbury:2018] for all intensive calculations, and these calculations are now performed
in single precision instead of double precision.  These two changes speed up the code by a factor
of ~60 with negligible decrease in accuracy, enabling a 0.8--5.3 um transmission spectrum to be
computed at R=20k in 0.35 ms on a RTX 5090.  To prevent underflows or overflows in intermediate
calculations, we instructed Claude to rearrange operations and rescale values (notably opacities and
polarizabilities) to be near unity.

An unrelated change which contributes to the vast speedup is our switch to nautilus [@Lange:2023] as the
default nested sampling package.  Unlike classic nested sampling packages such as MultiNest [@Feroz:2009] or
dynesty [@Speagle:2020], nautilus trains small neural networks to draw the live set boundary instead of fitting
hyper-ellipsoids or other geometric objects.  In our experience, nautilus enables far more efficient
sampling at high dimensionality, in addition to more accurate identification of small (low
prior volume) but high likelihood modes.

In addition to the large speedup, we add a variety of features, as detailed in our documentation and
examples. For example, it is now easy for users to run a forward model with custom temperature and
abundance profiles.  We add support for two-limb
retrievals because JWST observations have revealed the importance of limb asymmetry in
gas giant transit spectra [@Fairnington:2026].  We also add support for non-isothermal limbs in transit spectra,
again motivated by the increasing precision of data.


# State of the field
As of March 2023, at least 50 retrieval codes existed in the exoplanet literature, of which 26
were open source [@MacDonaldBatalha:2023].  However, highly optimized GPU retrieval codes are much rarer.
We are only aware of one: BeAR [@Wang:2026], a successor to HELIOS-r2 [@Kitzmann:2020],
written in C++ and CUDA with a Python wrapper.

Because the main motivation of our PLATON 7 rewrite was speed, we benchmarked
it against 4 widely used retrieval codes: PLATON 6 [@Zhang:2025], POSEIDON 1.4 [@MacDonald:2017; @MacDonald:2023], TauREx 3.3 [@AlRefaie:2021], and BeAR 2 [@Wang:2026].  We ran
free retrievals with the same parameterization on the transit spectrum of TOI 6894b [@Zhang:2026], making
sure that the spectral resolution and number of atmospheric layers are also identical across all codes.
All benchmarks were run on a vast.ai virtual instance with a AMD Ryzen 9 7950X3D 16-core CPU, 45 GB of RAM,
a RTX 5090 GPU, and CUDA 13.0.  The CPU codes were run with 16 cores for POSEIDON and TauREx, and 10 cores
(the maximum that the memory allows for) for PLATON 7 with GPU use disabled.

| Code        | Time (ms) | GPU?|
| ------------|-----------|-----|
| PLATON 7    | 0.35      | Yes |
| BeAR        | 0.61      | Yes |
| PLATON 6    | 20        | Yes |
| TauREx-CUDA | 17        | Yes |
| PLATON 7    | 14        | No  |
| POSEIDON    | 45        | No  |
| TauREx      | 132       | No  |

The table above reports the total wall clock time divided by the total number of forward models, after
discarding the initialization and 5000 forward models' worth of burn-in.
PLATON 7 is substantially faster than other codes we tested, beating BeAR by 1.75x, PLATON 6 by 57x, and
CPU-based codes by around 2-3 orders of magnitude.  When not allowed to use the GPU, PLATON 7 transparently falls back
to the CPU and remains substantially faster than other CPU codes.

The speedup enabled by nautilus is hard to quantify because it is highly dependent on the specific
retrieval problem, but we have seen speedups over MultiNest of more than 80x for retrievals with many
parameters.  This is especially the case for retrievals where each limb is fit separately with a
non-isothermal T/P profile.


# Software design
PLATON was initially designed to be fast, open source, easy to install and use, and easy to understand.
To this end, we implemented it in pure Python and kept the functionality spartan, enabling us to keep the number
of lines to a minimum.  The idea was that users could easily hack the code to implement the functionality they
wanted.  This idea was successful, and researchers from other groups implemented free retrievals, patchy
clouds, and rocky surface emission long before they were included in the official PLATON.

With the advent of powerful AI coding assistants, we de-emphasized the last design goal--
ease of human understanding--in favor of the first, raw speed.  Our experience is that frontier LLMs like Fable
are fully capable of understanding the code and making changes to it.  As AI becomes more powerful,
we expect only a vanishingly small portion of PLATON users will read the code themselves, making human readability
much less important.  That said, we have not sacrificed this goal completely: PLATON 7 is still in pure Python,
retains largely the same structure as PLATON 6, and implements the same core algorithms.

PLATON 7 remains easy to install and use, even without AI assistance.  Its user-facing interface is nearly
identical to that of its predecessor.  We have added more example scripts and simplified the ones that already
exist in order to flatten the learning curve even further for humans and AIs alike.


# Research impact statement
PLATON is one of the oldest and most widely used open source retrieval codes.  
It has been used by more than 80 papers to
investigate everything from sulfur biosignatures on sub-Neptunes [@Tsai:2024]
to aerosols on warm sub-Saturns [@Estrela:2021].  The majority of these papers
do not involve the main developer.  Users span institutions and career stages,
and include undergraduate students in exoplanet courses.

PLATON 7 specifically has been used to analyze the impact of limb asymmetry on
the transit spectrum of WASP-94b [@Fairnington:2026].  We expect much wider adoption once the
code is officially released, as the 57x speedup over its precedessor enables
use cases that would otherwise be prohibitively expensive.


# Our vibecoding experience
We expect that the majority of new software will be written by AI.  Since
PLATON 7 is the first AI-written retrieval code that we are aware of, we share
our vibecoding experiences in the hopes that it will be useful for the
community.

Many years ago, the lead developer MZ aspired to make PLATON
auto-differentiable by basing it on JAX instead of cupy [@Okuta:2017], enabling
the use of sample-efficient Hamiltonian Monte Carlo methods.  However, he
dropped the project after realized that JAX is not a drop-in replacement for
cupy, and that adopting JAX would mean rewriting the entire codebase.

In April 2026, we asked Claude Code with the Opus 4.6 model to implement HMC
with JAX-computed gradients.  Opus first misunderstood our intent and
computed numerical gradients instead of rewriting the code to support autodiff.
After we clarified, it successfully rewrote the transit spectrum code, but
encountered numerical difficulties with the opaque cloud deck, which we
circumvented by adopting a continuous cloud prescription.  After more debugging,
we realized that HMC struggles with the posterior shapes typical in free retrievals.  We therefore
disabled autodiff and switched back to nested sampling, enabling a significant speedup over the old PLATON.

We then realized that a further speedup is possible by using fp32 instead of
fp64 precision.  Opus made the change and converted all of the
arrays to fp32, but reported no speedup.  Realizing this made no sense, we
examined the AI's progress reports and realized it inexplicably set jax_enable_x64 to True.
We set the variable to False and the code instantly
became many times faster, evaluating every forward model in only 1.5 ms.
Opus declared that the code was unusable because fp32 is insufficient precision,
but we told it to work around overflows and underflows by reordering operations
and rescaling variables to order unity.  The code then produced correct results
for our test case.  However, over the following months, we found other bugs
stemming from the lower precision, which we fixed one by one.

In September 2026, we started from scratch and prompted Fable 5 to rewrite all
of PLATON--not just a small part of the transit spectrum code--to use JAX.  We told it from the outset
to use single precision, set jax_enable_x64 to True, and prevent underflows or
overflows by rescaling variables to order unity and/or reordering operations.
Afterwards, we told it to optimize the code while ensuring that the unit
tests still pass.  After it was finished, we
asked it to ask Codex (GPT-5.6 Sol) to review its changes and fix any important issues.
We also triggered /code-review to have Claude review its own changes, then
prompted it to fix any important issues.

This time, Claude was able to perform the entire refactoring and optimization process with no apparent issues and minimal human guidance.  We have exercised
the code extensively and could not find any new bugs.  We attribute this to
a combination of the much stronger model (Fable 5 vs. Opus 4.6), more specific
initial instructions, and extensive code review by both GPT-5.6 Sol and Fable.
Our experience is that frontier LLMs already outperform nearly all, if not all,
astronomers at writing correct and fast code.


# AI usage disclosure
Fable 5 with the Claude Code harness did all of the code rewrite.  GPT-5.6 Sol
with the Codex harness reviewed the code for correctness.  More features were subsequently added by Opus 5.5.

No AI was used in the writing of this manuscript, except for finding and adding references.

# Acknowledgements

We acknowledge contributions from Arjun Savel and Vatsal Panwar during the development of PLATON.

MZ acknowledges funding from the Heising-Simons Foundation for his 51 Pegasi b postdoctoral fellowship.


# References