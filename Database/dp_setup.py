from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import random
import pandas as pd
import numpy as np

db = SQLAlchemy()

class SalesData(db.Model):
    __tablename__ = 'sales_data'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, nullable=False)
    product_id = db.Column(db.String(50), nullable=False)
    region = db.Column(db.String(50), nullable=False)
    sales_amount = db.Column(db.Float, nullable=False)
    units_sold = db.Column(db.Integer, nullable=False)
    customer_segment = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ChurnData(db.Model):
    __tablename__ = 'churn_data'
    
    id = db.Column(db.Integer, primary_key=True)
    customer_id = db.Column(db.String(50), nullable=False)
    signup_date = db.Column(db.DateTime, nullable=False)
    last_activity_date = db.Column(db.DateTime)
    churn_flag = db.Column(db.Boolean, default=False)
    monthly_spend = db.Column(db.Float)
    total_purchases = db.Column(db.Integer)
    customer_segment = db.Column(db.String(50))
    predicted_churn_risk = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class OperationalMetrics(db.Model):
    __tablename__ = 'operational_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, nullable=False)
    server_uptime = db.Column(db.Float)
    support_tickets = db.Column(db.Integer)
    avg_response_time = db.Column(db.Float)
    customer_satisfaction = db.Column(db.Float)
    system_load = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AIInsights(db.Model):
    __tablename__ = 'ai_insights'
    
    id = db.Column(db.Integer, primary_key=True)
    insight_type = db.Column(db.String(50))  # 'summary', 'prediction', 'recommendation'
    metric_type = db.Column(db.String(50))   # 'sales', 'churn', 'operations'
    insight_text = db.Column(db.Text)
    confidence_score = db.Column(db.Float)
    related_data = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("Database initialized successfully!")

def generate_sample_data():
    """Generate realistic sample data for demonstration"""
    
    # Generating sales data
    sales_records = []
    start_date = datetime(2024, 1, 1)
    
    for i in range(365):  # One year of data
        current_date = start_date + pd.Timedelta(days=i)
        for region in ['North', 'South', 'East', 'West']:
            for product in ['Premium', 'Standard', 'Basic']:
                # Creating realistic patterns with trends and seasonality
                base_sales = 1000 + i * 2  # Upward trend
                weekend_effect = 1.5 if current_date.weekday() >= 5 else 1.0
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365)  # Yearly seasonality
                
                sales = base_sales * weekend_effect * seasonal_factor * random.uniform(0.9, 1.1)
                
                record = SalesData(
                    date=current_date,
                    product_id=f"PROD_{product}",
                    region=region,
                    sales_amount=round(sales, 2),
                    units_sold=int(sales / random.uniform(50, 150)),
                    customer_segment=random.choice(['Enterprise', 'SMB', 'Startup'])
                )
                sales_records.append(record)
    
    # Generating churn data
    churn_records = []
    for i in range(1000):  # 1000 customers
        signup_date = start_date + pd.Timedelta(days=random.randint(0, 300))
        monthly_spend = random.uniform(100, 10000)
        total_purchases = random.randint(1, 50)
        
        # Calculating churn probability based on factors
        churn_probability = 0.1
        if monthly_spend < 500:
            churn_probability += 0.2
        if total_purchases < 5:
            churn_probability += 0.15
            
        churned = random.random() < churn_probability
        
        record = ChurnData(
            customer_id=f"CUST_{i:05d}",
            signup_date=signup_date,
            last_activity_date=signup_date + pd.Timedelta(days=random.randint(30, 180)) if churned else datetime.now(),
            churn_flag=churned,
            monthly_spend=monthly_spend,
            total_purchases=total_purchases,
            customer_segment=random.choice(['Enterprise', 'SMB', 'Startup']),
            predicted_churn_risk=churn_probability
        )
        churn_records.append(record)
    
    # Generating operational metrics
    ops_records = []
    for i in range(90):  # 90 days of ops data
        current_date = datetime.now() - pd.Timedelta(days=90-i)
        
        # Creating realistic operational patterns
        base_uptime = 99.9
        incident_probability = 0.05
        
        if random.random() < incident_probability:
            base_uptime -= random.uniform(0.5, 5)
        
        record = OperationalMetrics(
            date=current_date,
            server_uptime=base_uptime,
            support_tickets=random.randint(10, 200),
            avg_response_time=random.uniform(5, 120),
            customer_satisfaction=random.uniform(3.5, 5.0),
            system_load=random.uniform(20, 95)
        )
        ops_records.append(record)
    
    # Bulk insert
    db.session.bulk_save_objects(sales_records)
    db.session.bulk_save_objects(churn_records)
    db.session.bulk_save_objects(ops_records)
    db.session.commit()
    
    print(f"Generated {len(sales_records)} sales records")
    print(f"Generated {len(churn_records)} customer records")
    print(f"Generated {len(ops_records)} operational records")