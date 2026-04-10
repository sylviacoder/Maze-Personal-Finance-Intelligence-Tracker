{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "5a2dc487-9b7f-4908-9bfa-1ebe7df9b7f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import joblib\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from xgboost import XGBClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "b9f79253-88c9-4468-bd64-61500d8a701a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train Model Saved\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv('https://raw.githubusercontent.com/sylviacoder/Maze-Personal-Finance-Intelligence-Tracker/refs/heads/main/data/cleaned/cleaned_data.csv')\n",
    "\n",
    "X = df.drop(columns=['debt_duress', 'user_id', 'actual_savings', 'savings_goal_met'], errors='ignore')\n",
    "y = df['financial_stress_level']\n",
    "le= LabelEncoder()\n",
    "y = le.fit_transform(y)\n",
    "\n",
    "num_col = X.select_dtypes(exclude = ['object']).columns\n",
    "cat_col = X.select_dtypes(include = ['object']).columns\n",
    "\n",
    "preprocessor = ColumnTransformer(transformers=[\n",
    "    ('num', StandardScaler(), num_col),\n",
    "    ('cat', OneHotEncoder(handle_unknown= 'ignore'), cat_col)\n",
    "])\n",
    "\n",
    "train_model = Pipeline(steps=[\n",
    "     ('preprocessor', preprocessor),\n",
    "     ('model', XGBClassifier(n_estimators=200,learning_rate=0.05,max_depth=5,random_state=42,eval_metric='mlogloss'))\n",
    "])\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
    "train_model.fit(X_train, y_train)\n",
    "joblib.dump(train_model, 'financial_stress_pipeline.pkl')\n",
    "print(\"Train Model Saved\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6924aad2-7bec-4ae0-a76b-2de5d6ad09bf",
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
