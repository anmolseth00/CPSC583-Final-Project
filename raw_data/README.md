# raw_data Directory

This directory contains the raw data files required to run the various experiments and models in this repository. The data primarily come from the Stanford COVID-19 Vaccine dataset and additional repositories associated with the Eterna OpenVaccine and DegScore projects.

## Data Sources

1. **Stanford COVID-19 Vaccine Dataset**  
   Downloaded from the Kaggle competition:  
   [https://www.kaggle.com/c/stanford-covid-vaccine/data](https://www.kaggle.com/c/stanford-covid-vaccine/data)
    We used this command to download the dataset: `kaggle competitions download -c stanford-covid-vaccine`
    Files such as `train.json`, `test.json`, and others related to the challenge are placed here. These contain sequence, structure, and measured degradation properties at various nucleotide positions.

3. **Data from Eterna and DegScore (12x dataset)**  
   The extended “12x dataset” which provides additional preprocessed features and secondary structure predictions, can be found here:  
   [https://www.kaggle.com/datasets/shujun717/openvaccine-12x-dataset](https://www.kaggle.com/datasets/shujun717/openvaccine-12x-dataset)  
   These enriched datasets are augmented and could be used to facilitate more advanced model training. Unfortunately the data is not well organized and there was no way to connect sequence IDs back to original OpenVaccine sequences. It also wasn't possible to recover original RNA sequences. The idea is robust, but re-implementation is left as a future direction.

## Specific Experiment Conditions

The following table lists the detailed experiment conditions and their corresponding data repositories from the DegScore project. These experiments represent different biochemical conditions under which RNA degradation or structure mapping was measured:

| Experiment                                | Data Repository                                                 |
| ----------------------------------------- | --------------------------------------------------------------- |
| pH 10, Mg2+=10 mM, 24˚C, 1 day            | [RYOS1_MGPH_0000](https://rmdb.stanford.edu/detail/RYOS1_MGPH_0000) |
| pH 10, Mg2+=0 mM, 24˚C, 7 days            | [RYOS1_PH10_0000](https://rmdb.stanford.edu/detail/RYOS1_PH10_0000) |
| pH 7.2, Mg2+=10 mM, 50˚C, 1 day           | [RYOS1_MG50_0000](https://rmdb.stanford.edu/detail/RYOS1_MG50_0000) |
| pH 7.2, Mg2+=0 mM, 50˚C, 7 days           | [RYOS1_50C_0000](https://rmdb.stanford.edu/detail/RYOS1_50C_0000)   |
| SHAPE structure mapping                   | [SHAPE_RYOS_0620](https://rmdb.stanford.edu/detail/SHAPE_RYOS_0620) |

These references were used to understand the experimental context of the data and to ensure the models were aligned with the experimental conditions originally measured.

## Data Exploration

Dataset exploration can be found in `dataset_exploration.ipynb`

## Notes

- Ensure you have the correct directory structure and file naming conventions as per the scripts in this repository.
- Some files are large and may not be included directly in the repo due to size limitations. Check the instructions or scripts for where to download and place these files.
- For more details on how these datasets are processed and utilized, refer to the main project documentation and experiment-specific READMEs.
