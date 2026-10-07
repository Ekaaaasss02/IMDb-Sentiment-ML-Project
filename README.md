# IMDb Sentiment Analysis 

This project implements the methodology described in the reference paper **“Machine Learning based classification for Sentimental analysis of IMDb reviews”** by Chun-Liang Wu and Song-Ling Shin.

## Pipeline

IMDb reviews → text cleaning → vectorization → six classifiers → confusion matrix / precision / accuracy → comparison

### Models
1. Logistic Regression
2. Support Vector Machine (SVM)
3. Multinomial Naïve Bayes
4. Random Forest
5. Boosting (`GradientBoostingClassifier`)
6. Deep Neural Network (`MLPClassifier`)

### Vectorization
- Binary
- Word Count
- 3-gram features
- TF-IDF

The reference paper reports its best result as **90.6% accuracy for DNN + binary + 3-gram**. Your result may differ because this implementation is a reproducible implementation of the methodology, not a claim of exact reproduction.

## Dataset

Download the IMDb CSV dataset and place it here:

```text
data/IMDB_Dataset.csv
```

Expected columns:

```text
review,sentiment
```

The common IMDb dataset contains 50,000 labeled reviews.

## Installation

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Linux/macOS:
```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

## Train the models

For a practical comparison:

```bash
python src/train.py --data data/IMDB_Dataset.csv --output artifacts
```

For a faster demo run:

```bash
python src/train.py --data data/IMDB_Dataset.csv --output artifacts --sample 10000
```

`--sample 10000` uses a balanced subset when possible.

To train only selected experiments:

```bash
python src/train.py --data data/IMDB_Dataset.csv --output artifacts --models logistic svm naive_bayes
```

## Run the web demo

After training the DNN binary + 3-gram model:

```bash
streamlit run app.py
```

Enter a movie review and the application will predict **Positive** or **Negative**.
Example : "This movie was terrible. The story was boring and the acting was awful."
          "This movie was absolutely fantastic. The acting was brilliant and the story was amazing."

## Project structure

```text
IMDb_Sentiment_ML_Project/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── IMDB_Dataset.csv
├── artifacts/
│   └── ... generated after training
└── src/
    ├── preprocessing.py
    ├── models.py
    ├── train.py
    └── evaluate.py
```

## Important implementation note

The paper does not specify every engineering detail needed to reproduce the experiment exactly (for example, vocabulary-size limits and the exact dataset file format). Therefore this code keeps the paper's main methodology and reported model configurations, while making those practical choices explicit in the code.

## Expected output

Training creates:

- `results.csv` — model/vectorizer metrics
- confusion-matrix PNG files
- serialized model/vectorizer files
- `best_model.json`

The evaluation uses:
- positive precision
- negative precision
- accuracy
- confusion matrix


