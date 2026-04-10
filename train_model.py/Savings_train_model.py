{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "76ebc2bd-9344-4aa1-bfc9-5b3205e0a7bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import joblib\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from xgboost import XGBClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "79b6a075-401d-48d7-bb14-bb1658a7ce7f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Saved\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv('https://raw.githubusercontent.com/sylviacoder/Maze-Personal-Finance-Intelligence-Tracker/refs/heads/main/data/cleaned/cleaned_data.csv')\n",
    "X = df.drop(columns=['savings_goal_met'])\n",
    "y = df['savings_goal_met']\n",
    "\n",
    "categorical_col = X.select_dtypes(include=['object']).columns\n",
    "numerical_col = X.select_dtypes(exclude=['object']).columns\n",
    "preprocessor = ColumnTransformer([\n",
    "    ('num', StandardScaler(), numerical_col),\n",
    "    ('cat', OneHotEncoder(handle_unknown= 'ignore'), categorical_col)\n",
    "])\n",
    "model_train = Pipeline(steps=[\n",
    "    ('preprocessor', preprocessor),\n",
    "    ('model', XGBClassifier(n_estimators=200, learning_rate=0.05, scale_pos_weight = 10,max_depth=5,random_state=42,eval_metric='logloss'))\n",
    "])\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42, stratify = y)\n",
    "model_train.fit(X_train, y_train)\n",
    "joblib.dump(model_train, 'savings_train_model.pkl')\n",
    "print('Model Saved')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "424acdc1-2950-461e-a971-2afad3a2e677",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
