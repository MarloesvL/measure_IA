# MeasureIABox

::: measureia.MeasureIABox
    handler: python
    options:
      show_source: true
      members_order: source
      show_root_heading: true
      heading_level: 2
      # measure_galaxy_contributions and assign_jackknife_patches come from mixins, so they
      # need inherited_members; the explicit list keeps the internals of the other bases out.
      inherited_members: true
      members:
        - __init__
        - measure_xi_w
        - measure_xi_multipoles
        - measure_galaxy_contributions
        - assign_jackknife_patches
