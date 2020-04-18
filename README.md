# LIVES study (Bag-of-Word)

In order to better understand the telephoned-based lifestyle intervention
in LIVES study, NLP and Machine Learning approach were conducted on 323
transcripts.

This repo contains:

1. Read fidelity measure and behavior outcome tables
2. Preprocess transcripts
3. feature engineering with NLP
4. Modeling with linear machine learning models
5. Evaluation on development set
6. Prediction on test set

## Overview of Data

Two tables providing predicted labels:
1. Fidelity measures (323 transcripts with 62 features)
2. Behavior outcomes (173 patients with 50 features)

Example,

| Fidelity Measures | Behavioral Outcomes |
| --- | --- |
| availability             | tfat_pcal_1 |
| specific_review___1      | tfat_pcal_2 |
| specific_review___2      | bmi1        |
| specific_recall___1      | bmi2        |

## Reminder before Getting Started

This instruction doesn't include LIVES study data. Please prepare it
before moving to the next step. And remember to put the data in the 
right position which will be specify in the following steps so that 
the scripts can spot it.

### Prerequisites

```
python>=3.72
numpy>=1.15
scikit-learn>=0.20
```

### Building a Working Directory (Folder)

First of all, make a directory (folder) called **LIVES_study**.
Feel free to change the name, it won't affect the following 
analysis.

```
mkdir LIVES_study
```

And then download or move the data folder in this new made folder.
In the default setting of the script, the directory name would 
be **Full dataset of fidelity scored calls with outcome data**.
for the downloaded data folder, but feel free to change the name 
as long as you remember to change the path in ***process.py*** as 
well. Noted that the data folder should have a folder of transcripts,
a fidelity measure table, and a behavior outcome table.

```
mv Full dataset of fidelity scored calls with outcome data LIVES_study/ 
```

### Installing

Clone this repository in the new made directory (folder)

```
git clone https://github.com/AllenTiTaiWang/LIVES_BagofWord.git
```

### Check Waht We Have Now

There should be one **LIVES_study** folder containing two folder inside,
which are **LIVES_BagofWord** and **Full dataset of fidelity scored 
calls with outcome data**

## Pipeline

The whole process has already built that there is only one command
to train the model, and another one to test it. The following flow
chart shows what the code will do.

![alt text](https://github.com/AllenTiTaiWang/LIVES_BagofWord/blob/master/pics/flow_chart.png)

### Run the script

Fisrt of all, we need to train the model, and see how it performs
on development set. Here, I take `availability` for example.

```
python3 main.py train availability logistic
```

This will give you the analysis of using `both`, `participant`,
`coach` utterance, and `whole`, `the first half`, `the second half`
of transcripts as hyperparameter to train models. Pick the best
combination, and we are ready to use them for final prediction.

```
python3 main.py test availability -e both -l 2 -p head
```

