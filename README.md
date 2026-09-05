# MLOps Project — <STUDENT_ID>

A version-controlled house-price prediction pipeline built for MLOps Assignment 1.
Separates source code (`src/`), raw data (`data/`, git-ignored), and trained
model artifacts (`model/`, git-ignored) so the repository stays lightweight
and reproducible.

## Project Structure
```
├── data/                      # raw dataset (git-ignored, generate locally)
├── src/
│   └── train_<STUDENT_ID>.py  # training script
├── model/                     # trained model output (git-ignored)
├── generate_data.py           # creates data/dataset.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/mlops-project-<STUDENT_ID>.git
cd mlops-project-<STUDENT_ID>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Run

```bash
python generate_data.py                 # creates data/dataset.csv
python src/train_<STUDENT_ID>.py        # trains model, saves to model/
```

Expected output: a `model/house_price_model.pkl` file and a printed
validation RMSE in the terminal.