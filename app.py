from flask import Flask, jsonify, send_file, request, make_response
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

app = Flask(__name__)

df = None
df_released = None
model = None
features = ['age', 'contact_number', 'infection_order']
target = 'recovery_duration'

def load_and_process_data():
    global df, df_released, model
    model = None
    # load dataset if available, else create dummy data
    if os.path.exists('covid_dataset.csv'):
        try:
            df = pd.read_csv('covid_dataset.csv')
            print("Loaded covid_dataset.csv")
        except Exception as e:
            print("Failed to read covid_dataset.csv:", e)
            df = None
    if df is None:
        # dummy fallback
        df = pd.DataFrame({
            'sex': ['male', 'female'] * 100,
            'age': np.random.randint(20, 80, 200),
            'region': ['Seoul', 'Daegu'] * 100,
            'state': ['released'] * 200,
            'confirmed_date': pd.date_range('2020-01-01', periods=200),
            'released_date': pd.date_range('2020-01-10', periods=200),
            'birth_year': np.random.randint(1940, 2000, 200),
            'contact_number': np.random.randint(0, 5, 200),
            'infection_order': np.random.randint(1, 10, 200),
            'infection_reason': ['contact'] * 200
        })

    if 'confirmed_date' in df.columns:
        df['confirmed_date'] = pd.to_datetime(df['confirmed_date'], errors='coerce')
    if 'released_date' in df.columns:
        df['released_date'] = pd.to_datetime(df['released_date'], errors='coerce')

    if 'birth_year' in df.columns and 'age' not in df.columns:
        df['age'] = pd.Timestamp.now().year - df['birth_year']
    elif 'age' not in df.columns:
        df['age'] = np.nan

    if 'released_date' in df.columns and 'confirmed_date' in df.columns:
        df['recovery_duration'] = (df['released_date'] - df['confirmed_date']).dt.days
    else:
        df['recovery_duration'] = np.nan

    if 'state' in df.columns:
        df_released = df[df['state'].astype(str).str.lower() == 'released'].copy()
    else:
        df_released = df.copy()

    # Train model if sklearn available and enough data
    if SKLEARN_AVAILABLE:
        required_cols = set(features + [target])
        if required_cols.issubset(df_released.columns):
            train_data = df_released.dropna(subset=features + [target])
            if len(train_data) >= 10:
                X = train_data[features]
                y = train_data[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model_local = LinearRegression()
                model_local.fit(X_train, y_train)
                y_pred = model_local.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = float(np.sqrt(mse))
                print(f"Model trained: R2={r2_score(y_test, y_pred):.2f}, RMSE={rmse:.2f}")
                model = model_local
            else:
                print("Not enough data to train model.")
        else:
            print("Required columns missing for training.")
    else:
        print("sklearn not available; skipping model training.")

    globals()['df'] = df
    globals()['df_released'] = df_released
    globals()['model'] = model

def plot_to_response(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/api/health')
def health():
    return jsonify({
        'sklearn_available': SKLEARN_AVAILABLE,
        'model_trained': model is not None,
        'rows_total': int(len(df)) if df is not None else 0,
        'rows_released': int(len(df_released)) if df_released is not None else 0
    })

@app.route('/api/demographics')
def demographics():
    if df_released is None or df_released.empty:
        return jsonify({'error': 'No data available'}), 400
    result = {
        'gender': df_released['sex'].value_counts().to_dict() if 'sex' in df_released.columns else {},
        'age': {
            'mean': float(df_released['age'].mean()) if 'age' in df_released.columns else None,
            'min': int(df_released['age'].min()) if 'age' in df_released.columns else None,
            'max': int(df_released['age'].max()) if 'age' in df_released.columns else None
        },
        'region': df_released['region'].value_counts().head(3).to_dict() if 'region' in df_released.columns else {}
    }
    return jsonify(result)

@app.route('/api/outcomes')
def outcomes():
    if df_released is None or 'recovery_duration' not in df_released.columns:
        return jsonify({'error': 'No recovery data'}), 400
    recovery_data = df_released['recovery_duration'].dropna()
    return jsonify({
        'avg_recovery': float(recovery_data.mean()),
        'median_recovery': float(recovery_data.median()),
        'recovery_range': f"{int(recovery_data.min())}-{int(recovery_data.max())} days",
        'count': int(len(recovery_data))
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'error': 'Bad request: no JSON body'}), 400

        age = float(data.get('age', 40))
        contact_number = float(data.get('contact_number', 0))
        infection_order = float(data.get('infection_order', 1))

        if model is not None:
            features_data = [[age, contact_number, infection_order]]
            pred = float(model.predict(features_data)[0])
        else:
            pred = max(1.0, 7.0 + (age - 40) * 0.05 + contact_number * 0.5 + (infection_order - 1) * 0.2)

        prediction = round(pred, 1)
        return jsonify({
            'prediction': prediction,
            'predicted_recovery_days': prediction,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/plot/gender')
def plot_gender():
    if df_released is None or 'sex' not in df_released.columns:
        return make_response(('No data', 404))
    fig, ax = plt.subplots(figsize=(4,3))
    df_released['sex'].value_counts().plot.pie(ax=ax, autopct='%1.1f%%')
    ax.set_ylabel('')
    return plot_to_response(fig)

@app.route('/api/plot/age')
def plot_age():
    if df_released is None or 'age' not in df_released.columns:
        return make_response(('No data', 404))
    fig, ax = plt.subplots(figsize=(4,3))
    sns.histplot(df_released['age'].dropna(), bins=10, ax=ax)
    ax.set_xlabel('Age')
    return plot_to_response(fig)

@app.route('/api/plot/region')
def plot_region():
    if df_released is None or 'region' not in df_released.columns:
        return make_response(('No data', 404))
    fig, ax = plt.subplots(figsize=(4,3))
    df_released['region'].value_counts().head(10).plot.bar(ax=ax)
    return plot_to_response(fig)

@app.route('/api/plot/recovery')
def plot_recovery():
    if df_released is None or 'recovery_duration' not in df_released.columns:
        return make_response(('No data', 404))
    fig, ax = plt.subplots(figsize=(4,3))
    sns.histplot(df_released['recovery_duration'].dropna(), bins=10, ax=ax)
    ax.set_xlabel('Recovery days')
    return plot_to_response(fig)

@app.route('/data')
def data():
    if os.path.exists('covid_dataset.csv'):
        try:
            df_full = pd.read_csv('covid_dataset.csv')
            return jsonify(df_full.to_dict(orient='records'))
        except Exception:
            return jsonify({'error': 'Failed to read dataset'}), 500
    return jsonify({'error': 'covid_dataset.csv not found'}), 404

@app.route('/')
def home():
    if os.path.exists('index.html'):
        return send_file('index.html')
    return jsonify({'message': 'Create index.html or visit /api/health'}), 200

if __name__ == '__main__':
    load_and_process_data()
    app.run(host='0.0.0.0', port=5000, debug=True)
