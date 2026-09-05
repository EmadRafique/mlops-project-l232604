# MLOps Project — l232604

A version-controlled house-price prediction pipeline built for MLOps Assignment 1.
Separates source code (`src/`), raw data (`data/`, git-ignored), and trained
model artifacts (`model/`, git-ignored) so the repository stays lightweight
and reproducible.

## Project Structure
```
├── data/                      # raw dataset (git-ignored, generate locally)
├── src/
│   └── train_l232604.py  # training script
├── model/                     # trained model output (git-ignored)
├── generate_data.py           # creates data/dataset.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/mlops-project-l232604.git
cd mlops-project-l232604

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Run

```bash
python generate_data.py                 # creates data/dataset.csv
python src/train_l232604.py        # trains model, saves to model/
```

Expected output: a `model/house_price_model.pkl` file and a printed
validation RMSE in the terminal.