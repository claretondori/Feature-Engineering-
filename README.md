# Housing Price Prediction: Testing Different Encoding Methods

This project focuses on predicting house prices using Housing dataset and a **Linear Regression** model. The main goal is to test **seven different categorical encoding methods** to see how they affect the model's accuracy, data shape, and processing speed.

---

## Project Highlights

- **Data Cleaning & Visual Insights:** Found and dropped extreme outliers, checked for missing data trends, and verified mathematical rules before training the model.
- **Preventing Data Leakage:** Used an advanced 5-Fold out-of-fold loop for Target Encoding to stop the model from cheating and overfitting.
- **Target Fixing:** Applied a log transformation ($y = \log(1+x)$) to house prices to turn a skewed distribution into a clean, balanced bell curve.
- **Performance Benchmarks:** Tracked and compared training accuracy ($R^2$) and validation error rates (RMSLE) across all used methods.

---

## Data Cleaning Pipeline

Linear regression models require clean, normally distributed numbers to work well. Based on charts generated during data exploration, the dataset was cleaned using these four steps:

1. **Fixing Skewed Prices:** The original house prices were heavily skewed to the right because of a few multi-million dollar luxury estates. Applying a log transformation (`np.log1p()`) compressed this tail and pulled the data into a perfect, symmetrical bell curve.
2. **Removing Outliers:** Found a small group of unusual properties with massive living areas (over 4,000 square feet) that sold for abnormally low prices (under $300,000). These outliers pull the trendline out of place, so they were deleted from the dataset.
3. **Dropping Empty Columns:** Built a missing value chart and dropped any column missing more than 40% of its data rows (`PoolQC`, `MiscFeature`, `Alley`, and `Fence`). Trying to guess values for columns this empty adds too much bad noise.
4. **Filling Minor Gaps:** For columns with very low missing rates, empty text spaces were filled with a default 'None' flag, and missing numeric entries were filled with the median value of that column.

---

## The Encoding Methods Explained

Because machine learning models cannot read words, text columns must be transformed into numbers. Here is how each tested method works:

### 1. One-Hot Encoding (OHE)

- **How it works:** Creates a brand-new column for every unique category option and fills it with 1s and 0s. The first option column is deleted to prevent redundant data.
- **Why it matters:** Highly accurate for linear models because the engine can calculate a custom price weight for each separate column tag.
- **Drawback:** Expands the dataset horizontally to over 200 columns, which uses a lot of computer memory.

### 2. Ordinal and Label Encoding

- **How it works:** Converts text labels directly into sequential whole numbers (0, 1, 2, 3...) down a single column.
- **Why it matters:** Keeps the dataset compact, but forces an artificial mathematical order onto unranked descriptions. The model assumes category 3 is three times larger than category 1, which confuses the math and raises the error rate.

### 3. Feature Hashing

- **How it works:** Uses a mathematical formula to compress all text options into a small, fixed number of columns that you choose beforehand.
- **Why it matters:** Saves immense computer memory, but causes hash collisions where two completely different traits get mapped into the exact same slot, confusing the model.

### 4. Frequency Encoding: (Encoding categories with dataset statistics)

- **How it works:** Replaces each text value with its total percentage frequency within the dataset.
- **Why it matters:** Shows the model how common or rare an attribute is. The weakness is that if an expensive neighborhood and a cheap neighborhood appear the same number of times, they get the exact same score, hiding their price differences.

### 5. Target and K-Fold Target Encoding

- **How it works:** Replaces text words with the average house price of that group. **K-Fold Target Encoding** splits the data into 5 separate folds. It calculates category price averages using 4 of the folds and applies them onto the remaining 1 fold.
- **Why it matters:** Creates an excellent linear connection with house prices without increasing column widths. The out-of-fold technique completely blocks target data leakage, stopping the model from overfitting.

---

## Performance Results

Every method was fitted using Scikit-Learn's `LinearRegression()` engine. Performance tracks training variance explained ($R^2$ Accuracy) and testing validation error (RMSLE). Lower error scores mean better predictions.

| Encoding Method                          | Dataset Width (Columns) | Training Accuracy ($R^2$) | Validation Error (RMSLE) | Model Profile                              |
| :--------------------------------------- | :---------------------: | :-----------------------: | :----------------------: | :----------------------------------------- |
| **Strategy A: One-Hot Encoding**         |      ~200+ Columns      |          ~0.9120          |    **0.1145 (Best)**     | Highly Precise / Uses More Memory          |
| **Method 6: K-Fold Target Encoding**     |       75 Columns        |          ~0.8950          |   **0.1210 (Strong)**    | Highly Efficient / Safest from Overfitting |
| **Strategy C: Standard Target Encoding** |       75 Columns        |          ~0.9010          |          0.1380          | Vulnerable to Data Leakage                 |
| **Method 4: Frequency Encoding**         |       75 Columns        |          ~0.8140          |          0.1760          | Missing Key Price Distinctions             |
| **Strategy B: Ordinal Encoding**         |       75 Columns        |          ~0.8310          |          0.1810          | Distorts Unordered Text Columns            |
| **Method 1: Label Encoding**             |       75 Columns        |          ~0.8290          |          0.1830          | Distorts Unordered Text Columns            |
| **Method 3: Feature Hashing ($N=10$)**   |       10 Columns        |          ~0.7920          |          0.1980          | High Memory Savings / Loses Accuracy       |

---

## Conclusion

For predicting real estate prices using a linear model, the ideal data engineering pipeline should use a hybrid approach:

- **For simple text fields with few options** (like `CentralAir` or `MSZoning`), **One-Hot Encoding** is the best choice because it builds clear, highly accurate binary switches with minimal memory footprint.
- **For highly complex text fields with many options** (specifically **`Neighborhood`**), One-Hot Encoding creates too many columns. **K-Fold Target Encoding** is the best choice here because it compresses the complex categories into a single linear price trend, keeping the dataset compact and safe from data leakage.
