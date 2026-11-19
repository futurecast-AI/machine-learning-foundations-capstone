# Data provenance and attribution

- Creator: Hadi Fanaee-T.
- Dataset: Bike Sharing, UCI Machine Learning Repository.
- Landing page: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- DOI: https://doi.org/10.24432/C5W894
- Source ZIP: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip
- Input: `hour.csv`, extracted unchanged; 17,379 rows and 17 columns.
- CSV SHA256: `e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`.
- Original data: real Capital Bikeshare rental logs for Washington, D.C.,
  aggregated hourly for 2011–2012, with weather and calendar information.
- The source Readme and actual file agree on 17,379 rows; UCI's landing page
  currently lists 17,389. Computations use the included file.
- UCI lists a Creative Commons Attribution 4.0 license.
- The source Readme requests citation of Fanaee-T and Gama's associated article:
  https://doi.org/10.1007/s13748-013-0040-3 (online 2013; journal issue 2014).
- Source normalization of weather is retained; no claim that its original
  preparation can be reconstructed from the released CSV is made.
- Season codes are kept as categories; source descriptions differ, so no
  potentially incorrect season-name relabeling is applied.

`cnt` counts observed rentals, not unique people, demand unmet because of empty
stations, or a causal effect of weather. Missing clock hours are not assumed
to have zero rentals. The file contains neither station-level nor demographic
information adequate to audit geographic or demographic fairness.
