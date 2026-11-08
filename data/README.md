# Data

Raw dataset files are fetched by script, never committed to this repo.
This file records the exact sources, versions, and checksums so any reader
can fetch byte-identical inputs.

## KuaiRec (primary)

- Source: Zenodo record 18164998, file `KuaiRec.zip` (data format v2.0,
  dated 2022-05-16 in the project README).
- URL: https://zenodo.org/records/18164998/files/KuaiRec.zip
- Size: 431,964,858 bytes.
- MD5 (as published on the Zenodo record): 261550d472c48eff4990fb13c0e5bcf7.
  The downloaded file's MD5 matches.
- SHA-256: c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c
- Files used: `data/big_matrix.csv`, `data/small_matrix.csv` from the
  archive. The feature files (including `user_features_raw.csv`) are not
  used; the design excludes them on privacy grounds.
- Licence: the archive ships a CC BY-SA 4.0 licence text; the Zenodo
  record metadata states CC BY 4.0; the GitHub README badge states
  CC BY-SA 4.0. This project uses the data under CC BY-SA 4.0 with
  attribution (see DECISIONS.md entry D3). Citation: Gao et al., KuaiRec,
  CIKM 2022, doi:10.1145/3511808.3557220.
- Fetch:
  `curl -sL https://zenodo.org/records/18164998/files/KuaiRec.zip -o data/raw/KuaiRec.zip`
  then unzip `KuaiRec 2.0/data/big_matrix.csv` and
  `KuaiRec 2.0/data/small_matrix.csv` into `data/raw/`.

## MovieLens-1M (second dataset, replication role)

- Source: GroupLens, https://files.grouplens.org/datasets/movielens/ml-1m.zip
- Size: 5,917,549 bytes.
- SHA-256: a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20
- File used: `ml-1m/ratings.dat`.
- Terms: GroupLens research use; redistribution is not permitted. Only
  derived statistics are published in this repo.
- Fetch:
  `curl -sL https://files.grouplens.org/datasets/movielens/ml-1m.zip -o data/raw/ml-1m.zip`
  then unzip into `data/raw/`.

## Derived data

Everything under `results/` is derived statistics computed by
`experiments/run_all.sh` from the inputs above. Derived data files are
covered by DATA_LICENSE.md (CC BY-SA 4.0, inherited from KuaiRec) where
they describe KuaiRec, and are published as statistics only for
MovieLens-1M.
