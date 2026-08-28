# MeasureIALightcone

::: measureia.MeasureIALightcone
    handler: python
    options:
      show_source: true
      members_order: source
      show_root_heading: true
      heading_level: 2
      # assign_jackknife_patches is inherited from MeasureJackknife; the explicit list keeps
      # the internals of the other bases out.
      inherited_members: true
      members:
        - __init__
        - measure_xi_w
        - measure_xi_multipoles
        - assign_jackknife_patches
