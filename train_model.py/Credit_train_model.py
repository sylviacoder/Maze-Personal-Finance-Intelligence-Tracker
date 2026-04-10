{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "99288abc-7258-4167-a2ee-75b1149fbfd8",
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
   "execution_count": 23,
   "id": "fba9652f-3cd2-4aee-907a-6a7f347f01cd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['credit_train_model.pkl']"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv(r'C:\\Users\\obian\\OneDrive\\Documents\\SJ Lytix Projects\\Maze\\Cleaned_Credit_Data.csv')\n",
    "df = df.drop(columns=['user_id', 'actual_savings', 'savings_goal_met','credit_score','debt_to_income_ratio'])\n",
    "\n",
    "X = df.drop(columns=['credit_risk'])\n",
    "y = df['credit_risk']\n",
    "le = LabelEncoder()\n",
    "y = le.fit_transform(y)\n",
    "\n",
    "categorical_cols = X.select_dtypes(include=['object']).columns\n",
    "numerical_cols= X.select_dtypes(exclude=['object']).columns\n",
    "\n",
    "preprocessor= ColumnTransformer(transformers=[\n",
    "    ('num', StandardScaler(), numerical_cols),\n",
    "    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)\n",
    "])\n",
    "train_model= Pipeline(steps= [\n",
    "    ('preprocessor', preprocessor),\n",
    "    ('model', XGBClassifier(n_estimators=200, max_depth=5, random_state=42, learning_rate=0.1, eval_metric='mlogloss'))\n",
    "])\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 42, stratify = y)\n",
    "train_model.fit(X_train, y_train)\n",
    "joblib.dump(train_model, 'credit_train_model.pkl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a9d0e2e6-30a5-4dc2-a5d4-9a1559656e54",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7d265c03-b2ad-41fd-b3d7-d22fcb43d062",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4c8e19f1-404c-42c4-b490-e6360c4a013b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e0c42d0a-bac6-4504-b697-93bd434a001f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "44e02e78-3482-4be2-85df-6e827bd8c697",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5d47dd8b-294a-422c-875d-62fb6bee54a2",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb8a66f8-9f17-4b70-a0d9-882f1704bd94",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "924b8861-e5f3-4a76-a89b-c5e0bc0f3a19",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c6553b0-998f-4773-8b30-074b0951201e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4b43535b-1939-4850-9f86-2bb1e12a8f26",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc57ee0c-95fe-464c-a483-e3b37e5f3fb4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1c56a08b-1716-4b25-8ac9-25236f519b74",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e1781055-cf09-4e57-81c9-513f30999fbc",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "50f90f93-f443-4e13-ac45-b102bb20f47d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb277205-949d-4acb-9709-55917d1a4902",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2843af93-758e-4ce0-bf53-852c0abd3223",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2edcab46-f004-48e6-95ed-643f1b481dd3",
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
