# Cleaned Data Dictionary

Both cleaned files are stored in `outputs/cleaned_data/`. Column names use
`snake_case`; no values have been imputed.

## Common columns

| Column | Type | Description |
|---|---|---|
| `region` | text | Indian state or union territory as named in the source file. |
| `date` | date (YYYY-MM-DD) | Month-end observation date parsed from the source. |
| `frequency` | text | Standardised to `Monthly` (`Monthly` and `M` both mapped). |
| `unemployment_rate` | float | Estimated unemployment rate (% of the labour force). |
| `estimated_employed` | float | Estimated number of employed people. |
| `labour_participation_rate` | float | Estimated labour participation rate (%). |
| `month` | text (YYYY-MM) | Calendar month key for grouping. |
| `month_start` | date | First day of the month, used for time axes. |
| `year` | integer | Calendar year. |
| `month_name` | text | Three-letter month abbreviation. |
| `month_number` | integer | Calendar month number (1-12). |
| `period` | text | `Pre-COVID` (before Mar 2020), `Transition` (Mar 2020), or `COVID period` (Apr 2020 onward). |
| `estimated_labour_force` | float | Derived as `employed / (1 - unemployment_rate/100)`. |
| `estimated_unemployed` | float | Derived as `estimated_labour_force - estimated_employed`. |
| `zero_unemployment` | boolean | `True` where the source reports an exact 0% rate; retained and flagged, not removed. |

## Dataset 1 only

| Column | Type | Description |
|---|---|---|
| `area` | text | `Rural` or `Urban`. |

File: `unemployment_india_clean.csv` (740 rows, 28 regions, 14 months).

## Dataset 2 only

| Column | Type | Description |
|---|---|---|
| `zone` | text | Geographic grouping from the source (`North`, `South`, `East`, `West`, `Northeast`). |
| `longitude` | float | State centroid longitude supplied by the source. |
| `latitude` | float | State centroid latitude supplied by the source. |

File: `unemployment_upto_11_2020_clean.csv` (267 rows, 27 regions, 10 months).

## Important measurement note

`estimated_labour_force` and `estimated_unemployed` are arithmetic derivations,
not new official estimates. They are used to calculate labour-force-weighted
aggregates so that large states are not given the same weight as small ones.
The source does not provide population weights or survey design details.
