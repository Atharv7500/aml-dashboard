import dash
from dash import html, dcc, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imbalanced_learn.over_sampling import SMOTE
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import mlflow
import mlflow.sklearn
from prometheus_client import start_http_server, Counter
import sentry_sdk
from faker import Faker
import psycopg2
import os
from fastapi import FastAPI
import uvicorn
from threading import Thread

# Mock integrations for other technologies (comments for those not runnable in prototype)
# Neo4j: Would connect to graph database for production graph queries/visualization (mocked with NetworkX)
# React: Frontend would be built with React + antd/Material UI for full UI (Dash used here as Python alternative)
# Airflow: Pipelines for ETL (mocked with in-code data processing)
# PostgreSQL: Storage (used for training data)
# Kafka/RabbitMQ: Message streaming (mocked with static data)
# Docker/Kubernetes: Containerization/deployment (run locally)
# Prometheus/Grafana: Monitoring (prometheus_client used for metrics)
# Sentry: Error tracking (sentry_sdk initialized)

# Initialize Sentry for error tracking
sentry_sdk.init(dsn="https://example@example.ingest.sentry.io/example")  # Replace with real DSN for production

# Initialize Prometheus metrics
alert_counter = Counter('aml_alerts_total', 'Total AML alerts generated')

# FastAPI integration (simple endpoint for API demo)
fastapi_app = FastAPI()

@fastapi_app.get("/api/alerts")
def get_alerts():
    return {"alerts": mock_data.to_dict('records')}

# Mock data generation (simulating ETL with pandas) with synthetic data
fake = Faker()
np.random.seed(42)
mock_data = pd.DataFrame({
    'transaction_id': range(1000),
    'amount': np.random.uniform(100, 10000, 1000),
    'currency': np.random.choice(['USD', 'EUR', 'BTC'], 1000),
    'risk_score': np.random.uniform(0, 1, 1000),
    'alert': np.random.choice([0, 1], 1000, p=[0.8, 0.2]),  # Imbalanced for demo
    'customer_id': np.random.randint(1, 200, 1000),
    'time_of_day': np.random.randint(0, 24, 1000),
    'customer_name': [fake.name() for _ in range(1000)],
    'transaction_description': [fake.sentence() for _ in range(1000)],
    'location': [fake.city() for _ in range(1000)],
    'date': [fake.date_this_year() for _ in range(1000)]
})

# Feature engineering (tabular features)
mock_data['amount_log'] = np.log(mock_data['amount'] + 1)
mock_data['is_high_amount'] = (mock_data['amount'] > 5000).astype(int)

# PostgreSQL integration for training data
db_url = os.environ.get('DATABASE_URL') or "postgresql://postgres:password@localhost/aml_db"
try:
    if not os.environ.get('DATABASE_URL'):
        # Create database if not exists (for local)
        conn_temp = psycopg2.connect(dbname="postgres", user="postgres", password="password", host="localhost")
        conn_temp.autocommit = True
        cur_temp = conn_temp.cursor()
        cur_temp.execute("SELECT 1 FROM pg_database WHERE datname = 'aml_db'")
        if not cur_temp.fetchone():
            cur_temp.execute("CREATE DATABASE aml_db")
        cur_temp.close()
        conn_temp.close()

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            transaction_id INT,
            amount FLOAT,
            currency TEXT,
            risk_score FLOAT,
            alert INT,
            customer_id INT,
            time_of_day INT,
            customer_name TEXT,
            transaction_description TEXT,
            location TEXT,
            date DATE,
            amount_log FLOAT,
            is_high_amount INT
        )
    """)
    # Check if data already exists
    cur.execute("SELECT COUNT(*) FROM transactions")
    if cur.fetchone()[0] == 0:
        for _, row in mock_data.iterrows():
            cur.execute("""
                INSERT INTO transactions (
                    transaction_id, amount, currency, risk_score, alert, customer_id, time_of_day,
                    customer_name, transaction_description, location, date, amount_log, is_high_amount
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, tuple(row))
    conn.commit()
    # Query data for training
    cur.execute("SELECT amount_log, is_high_amount, time_of_day, alert FROM transactions")
    data_from_db = cur.fetchall()
    X = [row[:-1] for row in data_from_db]
    y = [row[-1] for row in data_from_db]
    cur.close()
    conn.close()
except Exception as e:
    print(f"PostgreSQL connection failed: {e}. Using in-memory data for training.")
    X = mock_data[['amount_log', 'is_high_amount', 'time_of_day']].values.tolist()
    y = mock_data['alert'].tolist()

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train models
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)

lgb_model = LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)

# Log experiments with MLflow
mlflow.set_experiment("AML_Models")
with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", rf_model.score(X_test, y_test))
    mlflow.sklearn.log_model(rf_model, "model")

# Graph construction with NetworkX (prototyping)
G = nx.Graph()
customers = mock_data['customer_id'].unique()
for cust in customers:
    G.add_node(cust)
# Add edges based on shared transactions (simplified)
for i in range(len(mock_data)-1):
    if mock_data.iloc[i]['customer_id'] != mock_data.iloc[i+1]['customer_id']:
        G.add_edge(mock_data.iloc[i]['customer_id'], mock_data.iloc[i+1]['customer_id'])

# Mock GNN with PyTorch Geometric (basic example)
# Convert NetworkX to PyG Data (simplified)
node_features = torch.randn(len(customers), 10)  # Random features
edge_index = torch.tensor(list(G.edges())).t().contiguous()
data = Data(x=node_features, edge_index=edge_index)

class SimpleGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(10, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

gnn_model = SimpleGNN()
# Mock forward pass
out = gnn_model(data)

# Dash app for dashboard
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("AML Dashboard Prototype with Synthetic Dataset and PostgreSQL Training", style={'textAlign': 'center'}),
    html.P("Incorporating technologies: Python, pandas, numpy, plotly, scikit-learn, XGBoost, LightGBM, imbalanced-learn, NetworkX, PyTorch/PyG, FastAPI, MLflow, Prometheus, Sentry, Faker, PostgreSQL. (Others mocked/commented for prototype.)"),
    dcc.Tabs([
        dcc.Tab(label='Alerts & Risk Scores', children=[
            dcc.Graph(
                id='risk-bar',
                figure=px.bar(mock_data, x='transaction_id', y='risk_score', color='alert', title='Risk Scores by Transaction')
            ),
            dcc.Graph(
                id='scatter-plot',
                figure=px.scatter(mock_data, x='amount', y='risk_score', color='alert', title='Amount vs Risk Score')
            ),
            html.Div([
                html.H3("Model Predictions"),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[
                        {'label': 'Random Forest', 'value': 'rf'},
                        {'label': 'XGBoost', 'value': 'xgb'},
                        {'label': 'LightGBM', 'value': 'lgb'}
                    ],
                    value='rf'
                ),
                html.Div(id='model-output')
            ]),
            dcc.DataTable(
                id='data-table',
                columns=[{"name": i, "id": i} for i in mock_data.columns],
                data=mock_data.to_dict('records'),
                page_size=10
            )
        ]),
        dcc.Tab(label='Graph Analytics', children=[
            dcc.Graph(
                figure=go.Figure(data=[go.Scatter(x=[n[0] for n in G.nodes()], y=[n[1] for n in G.nodes()], mode='markers')]),
                # Simplified graph viz; use Neo4j Bloom or D3.js for production
            ),
            html.P("Graph constructed with NetworkX. GNN output shape: " + str(out.shape))
        ]),
        dcc.Tab(label='API & Monitoring', children=[
            html.P("FastAPI endpoint: /api/alerts (run separately for full API)"),
            html.P("Prometheus metrics initialized. Grafana dashboard would visualize in production."),
            html.Button("Trigger Alert", id='alert-button'),
            html.Div(id='alert-count')
        ])
    ])
])

@app.callback(
    Output('model-output', 'children'),
    Input('model-dropdown', 'value')
)
def update_model_output(model):
    if model == 'rf':
        pred = rf_model.predict(X_test)
    elif model == 'xgb':
        pred = xgb_model.predict(X_test)
    else:
        pred = lgb_model.predict(X_test)
    report = classification_report(y_test, pred, output_dict=True)
    return f"Precision: {report['1']['precision']:.2f}, Recall: {report['1']['recall']:.2f}"

@app.callback(
    Output('alert-count', 'children'),
    Input('alert-button', 'n_clicks')
)
def update_alert(n_clicks):
    if n_clicks:
        alert_counter.inc()
        return f"Alerts triggered: {n_clicks}"
    return "No alerts yet"

# Function to run FastAPI in thread (for demo)
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

if __name__ == '__main__':
    # Start Prometheus server
    start_http_server(8001)
    # Start FastAPI in thread
    Thread(target=run_fastapi).start()
    # Run Dash
    app.run_server(debug=True, host='0.0.0.0', port=8050)
