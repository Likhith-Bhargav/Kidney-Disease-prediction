from django.shortcuts import redirect, render
from django.http import HttpResponse
import pandas as pd
import numpy as np
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def home(request):
    return render(request, "prediction/index.html")

def medlibrary_view(request):
    return render(request, 'prediction/medlib.html')

def predict(request):
    if request.method == "POST":
        warnings.filterwarnings('ignore')
        df = pd.read_csv('venv/kidney_disease (2).csv')
        df.drop('id', axis=1, inplace=True)
        df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
                      'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                      'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
                      'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
                      'aanemia', 'class']

        df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
        df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
        df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')

        cat_cols = [col for col in df.columns if df[col].dtype == 'object']
        num_cols = [col for col in df.columns if df[col].dtype != 'object']

        df['diabetes_mellitus'].replace(to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)
        df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace='\tno', value='no')
        df['class'] = df['class'].replace(to_replace={'ckd\t': 'ckd', 'notckd': 'not ckd'})

        def random_value_imputation(feature):
            random_sample = df[feature].dropna().sample(df[feature].isna().sum())
            random_sample.index = df[df[feature].isnull()].index
            df.loc[df[feature].isnull(), feature] = random_sample

        def impute_mode(feature):
            mode = df[feature].mode()[0]
            df[feature] = df[feature].fillna(mode)

        for col in num_cols:
            random_value_imputation(col)

        random_value_imputation('red_blood_cells')
        random_value_imputation('pus_cell')

        for col in cat_cols:
            impute_mode(col)

        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col])

        ind_col = [col for col in df.columns if col != 'class']
        dep_col = 'class'

        X = df[ind_col]
        y = df[dep_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

        name = request.POST.get('name')
        age = float(request.POST.get('age', 0))
        blood_pressure = float(request.POST.get('bp', 0))
        specific_gravity = float(request.POST.get('sg', 0))
        albumin = float(request.POST.get('albumin', 0))
        sugar = float(request.POST.get('sugar', 0))
        red_blood_cells = int(request.POST.get('rbc', 0))
        pus_cell = int(request.POST.get('pc', 0))
        pus_cell_clumps = int(request.POST.get('pcc', 0))
        bacteria = int(request.POST.get('bacteria', 0))
        blood_glucose_random = float(request.POST.get('bgr', 0))
        blood_urea = float(request.POST.get('bu', 0))
        serum_creatinine = float(request.POST.get('sc', 0))
        sodium = float(request.POST.get('sodium', 0))
        potassium = float(request.POST.get('potassium', 0))
        haemoglobin = float(request.POST.get('haemoglobin', 0))
        packed_cell_volume = float(request.POST.get('pcv', 0))
        white_blood_cell_count = float(request.POST.get('wbc', 0))
        red_blood_cell_count = float(request.POST.get('rbc', 0))
        hypertension = int(request.POST.get('hypertension', 0))
        diabetes_mellitus = int(request.POST.get('dm', 0))
        coronary_artery_disease = int(request.POST.get('cad', 0))
        appetite = int(request.POST.get('appetite', 0))
        peda_edema = int(request.POST.get('pe', 0))
        aanemia = int(request.POST.get('aanemia', 0))

        models = {
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=0),
            'SVM': SVC(kernel='linear', random_state=0),
            'Naive Bayes': GaussianNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Extra Trees': ExtraTreesClassifier(),
            'XGBoost': XGBClassifier(objective='binary:logistic', learning_rate=0.5, max_depth=5, n_estimators=150),
            'Gradient Boost': GradientBoostingClassifier(),
            'AdaBoost': CatBoostClassifier(iterations=100, verbose=0)
        }

        # Train models
        for model in models.values():
            model.fit(X_train, y_train)

        def preprocess_input(data):
            return pd.DataFrame(data, index=[0])

        def classify_kidney_disease(model, input_data):
            processed_data = preprocess_input(input_data)
            prediction = model.predict(processed_data)
            return 'Kidney Disease' if prediction[0] == 0 else 'No Kidney Disease'

        input_data = {
            'age': age,
            'blood_pressure': blood_pressure,
            'specific_gravity': specific_gravity,
            'albumin': albumin,
            'sugar': sugar,
            'red_blood_cells': red_blood_cells,
            'pus_cell': pus_cell,
            'pus_cell_clumps': pus_cell_clumps,
            'bacteria': bacteria,
            'blood_glucose_random': blood_glucose_random,
            'blood_urea': blood_urea,
            'serum_creatinine': serum_creatinine,
            'sodium': sodium,
            'potassium': potassium,
            'haemoglobin': haemoglobin,
            'packed_cell_volume': packed_cell_volume,
            'white_blood_cell_count': white_blood_cell_count,
            'red_blood_cell_count': red_blood_cell_count,
            'hypertension': hypertension,
            'diabetes_mellitus': diabetes_mellitus,
            'coronary_artery_disease': coronary_artery_disease,
            'appetite': appetite,
            'peda_edema': peda_edema,
            'aanemia': aanemia,
        }

        results = {}
        for model_name, model in models.items():
            results[f'Result predicted by {model_name}'] = classify_kidney_disease(model, input_data)

        kidney_disease_count = sum(1 for result in results.values() if result == 'Kidney Disease')
        no_kidney_disease_count = len(results) - kidney_disease_count
        majority_prediction = 'Kidney Disease' if kidney_disease_count > no_kidney_disease_count else 'No Kidney Disease'

        request.session['results'] = results
        request.session['majority_prediction'] = majority_prediction

        return redirect('result')

    return render(request, "prediction/diseasePrediction.html")

def result(request):
    results = request.session.get('results', {})
    majority_prediction = request.session.get('majority_prediction', '')
    return render(request, 'prediction/result.html', {'results': results, 'majority_prediction': majority_prediction})