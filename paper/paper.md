---
title: 'MeasureIA: intrinsic alignment correlation functions for simulation boxes and lightcones'
tags:
  - Python
  - astronomy
  - cosmology
  - large-scale structure
  - intrinsic alignments
  - weak lensing
authors:
  - name: Marloes van Heukelum
    orcid: 0009-0008-3780-1617
    affiliation: 1
affiliations:
  - name: Utrecht University, The Netherlands
    index: 1
date: 11 August 2026
bibliography: paper.bib
---

<!--
  SKELETON. The front matter above is complete and JOSS-valid; the prose below is
  yours to write. Each section lists what JOSS reviewers look for, so you can
  delete the comment once the section is written.

  Length: JOSS papers are short — 250 to 1000 words in total is the norm. Aim for
  the low end; the documentation site carries the detail, and the paper only has to
  make a reader understand what the software is for and why it exists.

  Add co-authors by extending the `authors` and `affiliations` lists above; every
  author needs an ORCID, and affiliations are matched by `index`.
-->

# Summary

<!--
  One or two paragraphs, aimed at a *non-specialist* astronomer — someone who knows
  what a galaxy survey is but not what intrinsic alignments are. Cover:
    - what intrinsic alignments are, in a sentence, and why they matter (the main
      astrophysical contaminant of weak-lensing cosmology);
    - what the software measures: w_gg, w_g+ and the multipole moment estimator,
      with jackknife covariances;
    - the two data regimes it handles — periodic simulation boxes (cartesian,
      analytic randoms) and lightcones (sky coordinates, explicit random
      catalogue) — which is the distinguishing feature.
  Cite the multipole estimator you implement here.
-->

# Statement of need

<!--
  This is the section JOSS weighs most heavily. It should answer: who needs this,
  and what could they not do before?

  Points worth making, based on what the package actually does:
    - measuring IA correlations from hydrodynamic simulations usually means
      re-implementing estimators per project, with the shape and sign conventions
      (e+, ex, responsivity, the e1/e2 chirality) being a recurring source of
      error that is rarely written down;
    - existing tools cover parts of the problem but not the same ground: see the
      comparison below;
    - box and lightcone measurements normally require two different codes with
      different conventions, which makes simulation-to-survey comparison awkward.
      MeasureIA does both behind one interface with one set of documented
      conventions;
    - state the research context: what you (and your collaborators) use it for.
-->

# Functionality

<!--
  A brief tour. Suggested content:
    - the two entry points, `MeasureIABox` and `MeasureIALightcone`, and the data
      dictionaries they take;
    - the estimators: w_gg, w_g+ and the multipoles, with the "clusters" and
      "galaxies" lightcone variants;
    - jackknife covariances: sub-boxes for the box, k-means sky patches for the
      lightcone;
    - multiprocessing and the KD-tree paths for large catalogues;
    - HDF5 output with a documented structure, and `ReadData` to read it back;
    - `measureia.mocks`, which lets a reader reproduce any of it without data.
  A short code block helps; keep it to the ~10 lines of a real measurement.
-->

# Comparison to related software

<!--
  JOSS asks explicitly for this. Be concrete and fair — these are complementary
  tools, and MeasureIA is validated against all three, which is a strength worth
  stating.

    - halotools [@hearin2017]: mock_observables provides gi_plus_projected and wp
      for periodic boxes. Cover what MeasureIA adds (multipoles, jackknife
      covariance, lightcones) and note the agreement your validation reports.
    - TreeCorr [@jarvis2004]: fast, general-purpose correlations on the sky, but
      the IA estimator, its normalisation and the responsivity calibration have to
      be assembled by the user from raw counts.
    - corr_pc: independent implementation used for the multipole and jackknife
      cross-checks.

  Mention that the comparisons are shipped and runnable (validation/, and the
  reference-based tests), with the agreements quoted in the documentation. That
  is unusually strong evidence of correctness for a JOSS submission, so say so.
-->

# Acknowledgements

<!--
  Funding, grant numbers, collaborators, and anyone who tested the code or
  contributed conventions. Also acknowledge the packages MeasureIA is built on:
  NumPy [@harris2020], SciPy [@virtanen2020], Astropy [@astropy2022], h5py and
  CCL [@chisari2019].
-->

# References
