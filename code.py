# TASK 6 - NETFLIX CONTENT SUCCESS ANALYTICS ENGINE

# Machine Learning Internship Project
# Objective:
# Build an end-to-end machine learning analytics engine
# that analyzes Netflix content patterns and generates
# automated insights.
# Workflow:
# 1. Load and clean data
# 2. Perform advanced feature engineering
# 3. Analyze content patterns
# 4. Build multiple ML models
# 5. Compare model performance
# 6. Generate automated insights
# 7. Present findings through visual reports

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from IPython.display import display

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)


print("=" * 75)
print("TASK 6 - NETFLIX CONTENT SUCCESS ANALYTICS ENGINE")
print("=" * 75)

# 2. LOAD DATASET

df = pd.read_csv("Dataset.csv")

print("\nDataset loaded successfully!")

print("Dataset shape:", df.shape)

# 3. BASIC DATA EXPLORATION

print("\n" + "=" * 75)
print("DATASET INFORMATION")
print("=" * 75)

print("\nFirst 5 rows:")

display(df.head())


print("\nDataset columns:")

print(df.columns.tolist())

print("\nMissing values:")

print(df.isnull().sum())


print("\nDuplicate rows:")

print(df.duplicated().sum())

# 4. DATA CLEANING

df = df.drop_duplicates().copy()


categorical_columns = [
    "type",
    "director",
    "country",
    "rating",
    "duration",
    "listed_in"
]


for column in categorical_columns:

    df[column] = df[column].fillna(
        "Not Given"
    )


df["date_added"] = pd.to_datetime(
    df["date_added"],
    errors="coerce"
)


df["release_year"] = pd.to_numeric(
    df["release_year"],
    errors="coerce"
)


df["release_year"] = df[
    "release_year"
].fillna(
    df["release_year"].median()
)


print("\nAfter cleaning:")

print("Dataset shape:", df.shape)

# 5. ADVANCED FEATURE ENGINEERING

print("\n" + "=" * 75)
print("ADVANCED FEATURE ENGINEERING")
print("=" * 75)

# Feature 1: Content age

current_reference_year = 2026

df["content_age"] = (
    current_reference_year -
    df["release_year"]
)

# Feature 2: Date added year

df["added_year"] = (
    df["date_added"].dt.year
)

# Feature 3: Date added month

df["added_month"] = (
    df["date_added"].dt.month
)

# Feature 4: Genre count

df["genre_count"] = (
    df["listed_in"]
    .astype(str)
    .apply(
        lambda x:
        len(
            [
                item
                for item in x.split(",")
                if item.strip()
            ]
        )
    )
)

# Feature 5: Country count

df["country_count"] = (
    df["country"]
    .astype(str)
    .apply(
        lambda x:
        len(
            [
                item
                for item in x.split(",")
                if item.strip()
            ]
        )
    )
)

# Feature 6: Duration number

df["duration_number"] = (
    df["duration"]
    .astype(str)
    .str.extract(
        r"(\d+)"
    )[0]
)


df["duration_number"] = pd.to_numeric(
    df["duration_number"],
    errors="coerce"
)


df["duration_number"] = (
    df["duration_number"]
    .fillna(
        df["duration_number"].median()
    )
)

# Feature 7: Title length

df["title_length"] = (
    df["title"]
    .astype(str)
    .str.len()
)

# Feature 8: Number of words in title

df["title_word_count"] = (
    df["title"]
    .astype(str)
    .str.split()
    .str.len()
)


print("Created features:")

print("- content_age")
print("- added_year")
print("- added_month")
print("- genre_count")
print("- country_count")
print("- duration_number")
print("- title_length")
print("- title_word_count")

# 6. CONTENT OVERVIEW

print("\n" + "=" * 75)
print("CONTENT OVERVIEW")
print("=" * 75)


print(
    "\nTotal Netflix titles:",
    len(df)
)


print(
    "\nMovies:",
    int(
        (df["type"] == "Movie").sum()
    )
)


print(
    "TV Shows:",
    int(
        (df["type"] == "TV Show").sum()
    )
)


print(
    "\nUnique ratings:",
    df["rating"].nunique()
)


print(
    "Unique countries:",
    df["country"].nunique()
)

# 7. CONTENT TYPE ANALYSIS

content_type_counts = (
    df["type"]
    .value_counts()
)


print("\n" + "=" * 75)
print("CONTENT TYPE ANALYSIS")
print("=" * 75)


display(
    content_type_counts
)


plt.figure(figsize=(8, 6))

plt.bar(
    content_type_counts.index,
    content_type_counts.values
)

plt.title(
    "Netflix Content Type Distribution"
)

plt.xlabel("Content Type")

plt.ylabel("Number of Titles")

plt.tight_layout()

plt.show()

# 8. RATING ANALYSIS

rating_counts = (
    df["rating"]
    .value_counts()
    .head(15)
)


print("\n" + "=" * 75)
print("TOP NETFLIX RATINGS")
print("=" * 75)


display(
    rating_counts
)


plt.figure(figsize=(12, 6))

plt.bar(
    rating_counts.index,
    rating_counts.values
)

plt.title(
    "Top Netflix Audience Ratings"
)

plt.xlabel("Rating")

plt.ylabel("Number of Titles")

plt.xticks(
    rotation=45
)

plt.tight_layout()

plt.show()

# 9. TOP GENRE ANALYSIS

genre_counts = (
    df["listed_in"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(15)
)


print("\n" + "=" * 75)
print("TOP NETFLIX GENRES")
print("=" * 75)


display(
    genre_counts
)


plt.figure(figsize=(12, 6))

plt.bar(
    genre_counts.index,
    genre_counts.values
)

plt.title(
    "Top Netflix Genres"
)

plt.xlabel("Genre")

plt.ylabel("Number of Titles")

plt.xticks(
    rotation=60
)

plt.tight_layout()

plt.show()

# 10. CREATE ANALYTICS TARGET

# The original Netflix dataset does not contain a direct
# success/popularity score.
# Therefore, for this analytics engine we define a
# proxy success target based on content characteristics.
# A title is considered "Higher Analytics Score" when it
# has:
# - a newer release year
# - multiple genres
# - multiple countries
# The score is calculated from percentile ranks.
# This is an analytical proxy, NOT an actual Netflix
# popularity measurement.

df["release_recency_score"] = (
    df["release_year"].rank(
        pct=True
    )
)


df["genre_score"] = (
    df["genre_count"].rank(
        pct=True
    )
)


df["country_score"] = (
    df["country_count"].rank(
        pct=True
    )
)


df["analytics_score"] = (
    0.50 * df["release_recency_score"]
    +
    0.30 * df["genre_score"]
    +
    0.20 * df["country_score"]
)


# Define target using median

success_threshold = (
    df["analytics_score"].median()
)


df["success_target"] = (
    df["analytics_score"]
    >= success_threshold
).astype(int)


print("\n" + "=" * 75)
print("ANALYTICS TARGET CREATION")
print("=" * 75)


print(
    "Success threshold:",
    round(
        success_threshold,
        4
    )
)


print(
    "\nTarget distribution:"
)


print(
    df["success_target"]
    .value_counts()
)

# 11. TARGET DISTRIBUTION GRAPH

target_counts = (
    df["success_target"]
    .value_counts()
    .sort_index()
)


plt.figure(figsize=(8, 6))

plt.bar(
    [
        "Lower Analytics Score",
        "Higher Analytics Score"
    ],
    target_counts.values
)

plt.title(
    "Netflix Analytics Success Target Distribution"
)

plt.xlabel("Analytics Category")

plt.ylabel("Number of Titles")

plt.xticks(
    rotation=15
)

plt.tight_layout()

plt.show()

# 12. PREPARE MACHINE LEARNING DATA

features = [
    "type",
    "country",
    "rating",
    "release_year",
    "duration_number",
    "genre_count",
    "country_count",
    "content_age",
    "title_length",
    "title_word_count"
]


X = df[features].copy()

y = df["success_target"].copy()


print("\n" + "=" * 75)
print("MACHINE LEARNING FEATURES")
print("=" * 75)


print(
    "Features:"
)

for feature in features:

    print(
        "-",
        feature
    )

# 13. FEATURE TYPES

categorical_features = [
    "type",
    "country",
    "rating"
]


numerical_features = [
    "release_year",
    "duration_number",
    "genre_count",
    "country_count",
    "content_age",
    "title_length",
    "title_word_count"
]

# 14. PREPROCESSING PIPELINE

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        ),
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        )
    ]
)

# 15. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 75)
print("TRAIN-TEST SPLIT")
print("=" * 75)


print(
    "Training samples:",
    len(X_train)
)


print(
    "Testing samples:",
    len(X_test)
)

# 16. LOGISTIC REGRESSION

print("\n" + "=" * 75)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 75)


logistic_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000
            )
        )
    ]
)


logistic_model.fit(
    X_train,
    y_train
)


logistic_predictions = (
    logistic_model.predict(
        X_test
    )
)


logistic_probabilities = (
    logistic_model.predict_proba(
        X_test
    )[:, 1]
)

# 17. DECISION TREE

print("\n" + "=" * 75)
print("TRAINING DECISION TREE")
print("=" * 75)


decision_tree_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        )
    ]
)


decision_tree_model.fit(
    X_train,
    y_train
)


decision_tree_predictions = (
    decision_tree_model.predict(
        X_test
    )
)


decision_tree_probabilities = (
    decision_tree_model.predict_proba(
        X_test
    )[:, 1]
)

# 18. RANDOM FOREST

print("\n" + "=" * 75)
print("TRAINING RANDOM FOREST")
print("=" * 75)


random_forest_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


random_forest_model.fit(
    X_train,
    y_train
)


random_forest_predictions = (
    random_forest_model.predict(
        X_test
    )
)


random_forest_probabilities = (
    random_forest_model.predict_proba(
        X_test
    )[:, 1]
)

# 19. MODEL EVALUATION FUNCTION

def evaluate_model(
    model_name,
    y_true,
    predictions,
    probabilities
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_true,
        probabilities
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": auc
    }

# 20. COMPARE ALL MODELS

results = []


results.append(
    evaluate_model(
        "Logistic Regression",
        y_test,
        logistic_predictions,
        logistic_probabilities
    )
)


results.append(
    evaluate_model(
        "Decision Tree",
        y_test,
        decision_tree_predictions,
        decision_tree_probabilities
    )
)


results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        random_forest_predictions,
        random_forest_probabilities
    )
)


model_results = pd.DataFrame(
    results
)


print("\n" + "=" * 75)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 75)


display(
    model_results.round(4)
)

# 21. MODEL PERFORMANCE GRAPH

plt.figure(figsize=(11, 6))


x_positions = np.arange(
    len(model_results)
)


width = 0.18


plt.bar(
    x_positions - width,
    model_results["Accuracy"],
    width,
    label="Accuracy"
)


plt.bar(
    x_positions,
    model_results["Precision"],
    width,
    label="Precision"
)


plt.bar(
    x_positions + width,
    model_results["Recall"],
    width,
    label="Recall"
)


plt.bar(
    x_positions + 2 * width,
    model_results["F1_Score"],
    width,
    label="F1 Score"
)


plt.xticks(
    x_positions + width / 2,
    model_results["Model"],
    rotation=20
)


plt.ylabel(
    "Score"
)


plt.title(
    "Netflix Content Analytics Model Comparison"
)


plt.ylim(
    0,
    1.05
)


plt.legend()


plt.tight_layout()

plt.show()

# 22. RANDOM FOREST CLASSIFICATION REPORT

print("\n" + "=" * 75)
print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 75)


print(
    classification_report(
        y_test,
        random_forest_predictions,
        target_names=[
            "Lower Analytics Score",
            "Higher Analytics Score"
        ],
        zero_division=0
    )
)

# 23. RANDOM FOREST CONFUSION MATRIX

print("\n" + "=" * 75)
print("RANDOM FOREST CONFUSION MATRIX")
print("=" * 75)


cm = confusion_matrix(
    y_test,
    random_forest_predictions
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Lower",
        "Higher"
    ]
)


fig, ax = plt.subplots(
    figsize=(8, 7)
)


disp.plot(
    ax=ax,
    values_format="d"
)


plt.title(
    "Random Forest - Analytics Classification"
)


plt.tight_layout()

plt.show()

# 24. AUTOMATED INSIGHTS

print("\n" + "=" * 75)
print("AUTOMATED CONTENT INSIGHTS")
print("=" * 75)

# Insight 1 - Content type

most_common_type = (
    df["type"]
    .value_counts()
    .idxmax()
)


most_common_type_count = (
    df["type"]
    .value_counts()
    .max()
)


print(
    "\nInsight 1:"
)


print(
    f"{most_common_type} is the most common "
    f"content type with {most_common_type_count} titles."
)

# Insight 2 - Most common rating

most_common_rating = (
    df["rating"]
    .value_counts()
    .idxmax()
)


most_common_rating_count = (
    df["rating"]
    .value_counts()
    .max()
)


print(
    "\nInsight 2:"
)


print(
    f"{most_common_rating} is the most frequent "
    f"rating category with {most_common_rating_count} titles."
)

# Insight 3 - Most common genre

top_genre = (
    genre_counts
    .idxmax()
)


top_genre_count = (
    genre_counts
    .max()
)


print(
    "\nInsight 3:"
)


print(
    f"{top_genre} is the most common genre "
    f"with {top_genre_count} titles."
)

# Insight 4 - Average release year

average_release_year = (
    df["release_year"]
    .mean()
)


print(
    "\nInsight 4:"
)


print(
    f"The average release year of the "
    f"available Netflix content is "
    f"{average_release_year:.1f}."
)

# Insight 5 - Analytics categories

high_score_percentage = (
    df["success_target"]
    .mean()
) * 100


print(
    "\nInsight 5:"
)


print(
    f"{high_score_percentage:.2f}% of the dataset "
    f"falls into the Higher Analytics Score category "
    f"under the project-defined proxy."
)

# 25. FEATURE IMPORTANCE - RANDOM FOREST

print("\n" + "=" * 75)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 75)


rf_pipeline = random_forest_model


fitted_preprocessor = (
    rf_pipeline
    .named_steps[
        "preprocessor"
    ]
)


rf_classifier = (
    rf_pipeline
    .named_steps[
        "classifier"
    ]
)


feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


feature_importance = pd.DataFrame({

    "Feature":
        feature_names,

    "Importance":
        rf_classifier.feature_importances_

})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print(
    "\nTop 15 important features:"
)


display(
    feature_importance.head(15)
)

# 26. FEATURE IMPORTANCE GRAPH

top_features = (
    feature_importance
    .head(15)
    .sort_values(
        "Importance"
    )
)


plt.figure(figsize=(10, 7))


plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)


plt.title(
    "Top Random Forest Feature Importances"
)


plt.xlabel(
    "Importance"
)


plt.tight_layout()

plt.show()

# 27. ANALYTICS SCORE DISTRIBUTION

plt.figure(figsize=(10, 6))


plt.hist(
    df["analytics_score"],
    bins=30
)


plt.axvline(
    success_threshold,
    linestyle="--",
    label="Median Threshold"
)


plt.title(
    "Netflix Analytics Score Distribution"
)


plt.xlabel(
    "Analytics Score"
)


plt.ylabel(
    "Number of Titles"
)


plt.legend()


plt.tight_layout()

plt.show()

# 28. TOP ANALYTICS CONTENT

top_content = (
    df[
        [
            "title",
            "type",
            "rating",
            "release_year",
            "genre_count",
            "country_count",
            "analytics_score"
        ]
    ]
    .sort_values(
        "analytics_score",
        ascending=False
    )
    .head(15)
)


print("\n" + "=" * 75)
print("TOP CONTENT BY PROJECT ANALYTICS SCORE")
print("=" * 75)


display(
    top_content
)

# 29. CONTENT ANALYTICS BY TYPE

type_analysis = (
    df.groupby("type")
    .agg(
        Average_Analytics_Score=(
            "analytics_score",
            "mean"
        ),
        Average_Genres=(
            "genre_count",
            "mean"
        ),
        Average_Countries=(
            "country_count",
            "mean"
        ),
        Average_Release_Year=(
            "release_year",
            "mean"
        ),
        Number_of_Titles=(
            "title",
            "count"
        )
    )
    .round(3)
)


print("\n" + "=" * 75)
print("CONTENT ANALYTICS BY TYPE")
print("=" * 75)


display(
    type_analysis
)

# 30. CONTENT ANALYTICS BY RATING

rating_analysis = (
    df.groupby("rating")
    .agg(
        Average_Analytics_Score=(
            "analytics_score",
            "mean"
        ),
        Number_of_Titles=(
            "title",
            "count"
        )
    )
    .sort_values(
        "Average_Analytics_Score",
        ascending=False
    )
    .head(15)
    .round(3)
)


print("\n" + "=" * 75)
print("CONTENT ANALYTICS BY RATING")
print("=" * 75)


display(
    rating_analysis
)

# 31. CREATE FINAL ANALYTICS REPORT

analytics_report = pd.DataFrame({

    "Metric": [

        "Total Titles",

        "Movies",

        "TV Shows",

        "Unique Ratings",

        "Unique Countries",

        "Most Common Type",

        "Most Common Rating",

        "Top Genre",

        "Average Release Year",

        "High Analytics Score Percentage",

        "Best Model",

        "Best Model Accuracy"

    ],

    "Value": [

        len(df),

        int(
            (df["type"] == "Movie").sum()
        ),

        int(
            (df["type"] == "TV Show").sum()
        ),

        df["rating"].nunique(),

        df["country"].nunique(),

        most_common_type,

        most_common_rating,

        top_genre,

        round(
            average_release_year,
            2
        ),

        round(
            high_score_percentage,
            2
        ),

        model_results.loc[
            model_results[
                "Accuracy"
            ].idxmax(),
            "Model"
        ],

        round(
            model_results[
                "Accuracy"
            ].max(),
            4
        )

    ]

})


print("\n" + "=" * 75)
print("FINAL ANALYTICS REPORT")
print("=" * 75)


display(
    analytics_report
)

# 32. SAVE ANALYTICS DATA

df.to_csv(
    "netflix_content_analytics_dataset.csv",
    index=False
)

# 33. SAVE MODEL RESULTS

model_results.to_csv(
    "netflix_analytics_model_results.csv",
    index=False
)

# 34. SAVE FEATURE IMPORTANCE

feature_importance.to_csv(
    "netflix_feature_importance.csv",
    index=False
)

# 35. SAVE ANALYTICS REPORT

analytics_report.to_csv(
    "netflix_analytics_report.csv",
    index=False
)

# 36. SAVE TOP CONTENT

top_content.to_csv(
    "netflix_top_analytics_content.csv",
    index=False
)

# 37. SAVE TYPE ANALYSIS

type_analysis.to_csv(
    "netflix_type_analytics.csv"
)

# 38. SAVE RATING ANALYSIS

rating_analysis.to_csv(
    "netflix_rating_analytics.csv"
)

# 39. SAVE BEST MODEL

best_model_name = (
    model_results.loc[
        model_results[
            "Accuracy"
        ].idxmax(),
        "Model"
    ]
)


if best_model_name == "Logistic Regression":

    best_model = logistic_model

elif best_model_name == "Decision Tree":

    best_model = decision_tree_model

else:

    best_model = random_forest_model


joblib.dump(
    best_model,
    "netflix_content_success_model.pkl"
)

# 40. SAVE FEATURE PREPROCESSOR

joblib.dump(
    preprocessor,
    "netflix_content_analytics_preprocessor.pkl"
)

# 41. FINAL OUTPUT FILES

print("\n" + "=" * 75)
print("FILES SAVED SUCCESSFULLY")
print("=" * 75)


print(
    "1. netflix_content_analytics_dataset.csv"
)

print(
    "2. netflix_analytics_model_results.csv"
)

print(
    "3. netflix_feature_importance.csv"
)

print(
    "4. netflix_analytics_report.csv"
)

print(
    "5. netflix_top_analytics_content.csv"
)

print(
    "6. netflix_type_analytics.csv"
)

print(
    "7. netflix_rating_analytics.csv"
)

print(
    "8. netflix_content_success_model.pkl"
)

print(
    "9. netflix_content_analytics_preprocessor.pkl"
)

# 42. FINAL PROJECT SUMMARY

print("\n")
print("=" * 75)
print("TASK 6 - PROJECT SUMMARY")
print("=" * 75)


print(
    "Project:"
)


print(
    "Netflix Content Success Analytics Engine"
)


print(
    "\nTotal titles analyzed:",
    len(df)
)


print(
    "\nModels trained:"
)


print(
    "- Logistic Regression"
)

print(
    "- Decision Tree"
)

print(
    "- Random Forest"
)


print(
    "\nBest model:",
    best_model_name
)


print(
    "Best model accuracy:",
    round(
        model_results[
            "Accuracy"
        ].max(),
        4
    )
)


print(
    "\nTop genre:",
    top_genre
)


print(
    "Most common content type:",
    most_common_type
)


print(
    "Most common rating:",
    most_common_rating
)


print(
    "\nGenerated automated insights:",
    5
)


print(
    "\nAnalytics report generated successfully!"
)


print("\n" + "=" * 75)

print(
    "TASK 6 COMPLETED SUCCESSFULLY!"
)

print("=" * 75)

Output:
===========================================================================
TASK 6 - NETFLIX CONTENT SUCCESS ANALYTICS ENGINE
===========================================================================

Dataset loaded successfully!
Dataset shape: (8790, 10)

===========================================================================
DATASET INFORMATION
===========================================================================

First 5 rows:
show_id	type	title	director	country	date_added	release_year	rating	duration	listed_in
0	s1	Movie	Dick Johnson Is Dead	Kirsten Johnson	United States	9/25/2021	2020	PG-13	90 min	Documentaries
1	s3	TV Show	Ganglands	Julien Leclercq	France	9/24/2021	2021	TV-MA	1 Season	Crime TV Shows, International TV Shows, TV Act...
2	s6	TV Show	Midnight Mass	Mike Flanagan	United States	9/24/2021	2021	TV-MA	1 Season	TV Dramas, TV Horror, TV Mysteries
3	s14	Movie	Confessions of an Invisible Girl	Bruno Garotti	Brazil	9/22/2021	2021	TV-PG	91 min	Children & Family Movies, Comedies
4	s8	Movie	Sankofa	Haile Gerima	United States	9/24/2021	1993	TV-MA	125 min	Dramas, Independent Movies, International Movies

Dataset columns:
['show_id', 'type', 'title', 'director', 'country', 'date_added', 'release_year', 'rating', 'duration', 'listed_in']

Missing values:
show_id         0
type            0
title           0
director        0
country         0
date_added      0
release_year    0
rating          0
duration        0
listed_in       0
dtype: int64

Duplicate rows:
0

After cleaning:
Dataset shape: (8790, 10)

===========================================================================
ADVANCED FEATURE ENGINEERING
===========================================================================
Created features:
- content_age
- added_year
- added_month
- genre_count
- country_count
- duration_number
- title_length
- title_word_count

===========================================================================
CONTENT OVERVIEW
===========================================================================

Total Netflix titles: 8790

Movies: 6126
TV Shows: 2664

Unique ratings: 14
Unique countries: 86

===========================================================================
CONTENT TYPE ANALYSIS
===========================================================================
count
type	
Movie	6126
TV Show	2664

dtype: int64


===========================================================================
TOP NETFLIX RATINGS
===========================================================================
count
rating	
TV-MA	3205
TV-14	2157
TV-PG	861
R	799
PG-13	490
TV-Y7	333
TV-Y	306
PG	287
TV-G	220
NR	79
G	41
TV-Y7-FV	6
NC-17	3
UR	3

dtype: int64


===========================================================================
TOP NETFLIX GENRES
===========================================================================
count
listed_in	
International Movies	2752
Dramas	2426
Comedies	1674
International TV Shows	1349
Documentaries	869
Action & Adventure	859
TV Dramas	762
Independent Movies	756
Children & Family Movies	641
Romantic Movies	616
Thrillers	577
TV Comedies	573
Crime TV Shows	469
Kids' TV	448
Docuseries	394

dtype: int64


===========================================================================
ANALYTICS TARGET CREATION
===========================================================================
Success threshold: 0.4992

Target distribution:
success_target
1    4470
0    4320
Name: count, dtype: int64


===========================================================================
MACHINE LEARNING FEATURES
===========================================================================
Features:
- type
- country
- rating
- release_year
- duration_number
- genre_count
- country_count
- content_age
- title_length
- title_word_count

===========================================================================
TRAIN-TEST SPLIT
===========================================================================
Training samples: 7032
Testing samples: 1758

===========================================================================
TRAINING LOGISTIC REGRESSION
===========================================================================

===========================================================================
TRAINING DECISION TREE
===========================================================================

===========================================================================
TRAINING RANDOM FOREST
===========================================================================

===========================================================================
MODEL PERFORMANCE COMPARISON
===========================================================================
Model	Accuracy	Precision	Recall	F1_Score	ROC_AUC
0	Logistic Regression	0.9573	0.938	0.981	0.959	0.9961
1	Decision Tree	1.0000	1.000	1.000	1.000	1.0000
2	Random Forest	1.0000	1.000	1.000	1.000	1.0000


===========================================================================
RANDOM FOREST CLASSIFICATION REPORT
===========================================================================
                        precision    recall  f1-score   support

 Lower Analytics Score       1.00      1.00      1.00       864
Higher Analytics Score       1.00      1.00      1.00       894

              accuracy                           1.00      1758
             macro avg       1.00      1.00      1.00      1758
          weighted avg       1.00      1.00      1.00      1758


===========================================================================
RANDOM FOREST CONFUSION MATRIX
===========================================================================


===========================================================================
AUTOMATED CONTENT INSIGHTS
===========================================================================

Insight 1:
Movie is the most common content type with 6126 titles.

Insight 2:
TV-MA is the most frequent rating category with 3205 titles.

Insight 3:
International Movies is the most common genre with 2752 titles.

Insight 4:
The average release year of the available Netflix content is 2014.2.

Insight 5:
50.85% of the dataset falls into the Higher Analytics Score category under the project-defined proxy.

===========================================================================
RANDOM FOREST FEATURE IMPORTANCE
===========================================================================

Top 15 important features:
Feature	Importance
96	numerical__release_year	0.376790
100	numerical__content_age	0.330548
98	numerical__genre_count	0.149149
97	numerical__duration_number	0.026069
90	categorical__rating_TV-MA	0.016136
76	categorical__country_United States	0.012460
101	numerical__title_length	0.012360
0	categorical__type_Movie	0.010167
1	categorical__type_TV Show	0.008864
102	numerical__title_word_count	0.007284
87	categorical__rating_R	0.004608
86	categorical__rating_PG-13	0.003553
88	categorical__rating_TV-14	0.002733
28	categorical__country_India	0.002722
66	categorical__country_Spain	0.002563



===========================================================================
TOP CONTENT BY PROJECT ANALYTICS SCORE
===========================================================================
title	type	rating	release_year	genre_count	country_count	analytics_score
6920	Monarca	TV Show	TV-MA	2021	3	1	0.819636
6915	Cobra Kai	TV Show	TV-14	2021	3	1	0.819636
6913	Nailed It! Mexico	TV Show	TV-PG	2021	3	1	0.819636
27	Monsters Inside: The 24 Faces of Billy Milligan	TV Show	TV-14	2021	3	1	0.819636
24	Bangkok Breaking	TV Show	TV-MA	2021	3	1	0.819636
18	Crime Stories: India Detectives	TV Show	TV-MA	2021	3	1	0.819636
6910	Stuck Apart	Movie	TV-MA	2021	3	1	0.819636
6907	Inside the World’s Toughest Prisons	TV Show	TV-MA	2021	3	1	0.819636
6904	Disenchantment	TV Show	TV-14	2021	3	1	0.819636
6899	Daughter From Another Mother	TV Show	TV-MA	2021	3	1	0.819636
6896	Busted!	TV Show	TV-14	2021	3	1	0.819636
6891	50M2	TV Show	TV-MA	2021	3	1	0.819636
6878	Invisible City	TV Show	TV-MA	2021	3	1	0.819636
6877	Hache	TV Show	TV-MA	2021	3	1	0.819636
6871	Nadiya Bakes	TV Show	TV-G	2021	3	1	0.819636

===========================================================================
CONTENT ANALYTICS BY TYPE
===========================================================================
Average_Analytics_Score	Average_Genres	Average_Countries	Average_Release_Year	Number_of_Titles
type					
Movie	0.473	2.152	1.0	2013.120	6126
TV Show	0.562	2.293	1.0	2016.628	2664

===========================================================================
CONTENT ANALYTICS BY RATING
===========================================================================
Average_Analytics_Score	Number_of_Titles
rating		
TV-MA	0.558	3205
TV-G	0.529	220
TV-14	0.514	2157
TV-PG	0.492	861
TV-Y	0.468	306
NC-17	0.452	3
TV-Y7	0.451	333
TV-Y7-FV	0.422	6
R	0.400	799
PG	0.398	287
UR	0.389	3
NR	0.379	79
PG-13	0.378	490
G	0.294	41

===========================================================================
FINAL ANALYTICS REPORT
===========================================================================
Metric	Value
0	Total Titles	8790
1	Movies	6126
2	TV Shows	2664
3	Unique Ratings	14
4	Unique Countries	86
5	Most Common Type	Movie
6	Most Common Rating	TV-MA
7	Top Genre	International Movies
8	Average Release Year	2014.18
9	High Analytics Score Percentage	50.85
10	Best Model	Decision Tree
11	Best Model Accuracy	1.0

===========================================================================
FILES SAVED SUCCESSFULLY
===========================================================================
1. netflix_content_analytics_dataset.csv
2. netflix_analytics_model_results.csv
3. netflix_feature_importance.csv
4. netflix_analytics_report.csv
5. netflix_top_analytics_content.csv
6. netflix_type_analytics.csv
7. netflix_rating_analytics.csv
8. netflix_content_success_model.pkl
9. netflix_content_analytics_preprocessor.pkl


===========================================================================
TASK 6 - PROJECT SUMMARY
===========================================================================
Project:
Netflix Content Success Analytics Engine

Total titles analyzed: 8790

Models trained:
- Logistic Regression
- Decision Tree
- Random Forest
Best model: Decision Tree
Best model accuracy: 1.0

Top genre: International Movies
Most common content type: Movie
Most common rating: TV-MA

Generated automated insights: 5

Analytics report generated successfully!

===========================================================================
TASK 6 COMPLETED SUCCESSFULLY!
===========================================================================
