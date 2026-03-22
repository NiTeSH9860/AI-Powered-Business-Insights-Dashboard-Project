from flask import Blueprint, jsonify, request, render_template
from Database.dp_setup import db, SalesData, ChurnData, OperationalMetrics, AIInsights
from models.predictor import SalesPredictor, ChurnPredictor
from models.summarizer import BusinessSummarizer
from config import Config
from datetime import datetime, timedelta
import pandas as pd
import json

api_bp = Blueprint('api', __name__)

# Initializing AI components
summarizer = BusinessSummarizer(api_key=Config.OPENAI_API_KEY)
sales_predictor = SalesPredictor()
churn_predictor = ChurnPredictor()

@api_bp.route('/')
def index():
    """Render main dashboard"""
    return render_template('index.html')

@api_bp.route('/dashboard')
def dashboard():
    """Render dashboard view"""
    return render_template('dashboard.html')

@api_bp.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@api_bp.route('/api/metrics/current', methods=['GET'])
def get_current_metrics():
    """Getting current business metrics"""
    try:
        # Getting date range
        days = int(request.args.get('days', 30))
        start_date = datetime.now() - timedelta(days=days)
        
        # Query sales data
        sales_data = SalesData.query.filter(SalesData.date >= start_date).all()
        sales_df = pd.DataFrame([{
            'date': s.date,
            'amount': s.sales_amount,
            'region': s.region,
            'product': s.product_id
        } for s in sales_data])
        
        # Query churn data
        churn_data = ChurnData.query.all()
        churn_df = pd.DataFrame([{
            'customer_id': c.customer_id,
            'churn_flag': c.churn_flag,
            'monthly_spend': c.monthly_spend,
            'segment': c.customer_segment
        } for c in churn_data])
        
        # Query operational data
        ops_data = OperationalMetrics.query.filter(OperationalMetrics.date >= start_date).all()
        ops_df = pd.DataFrame([{
            'date': o.date,
            'uptime': o.server_uptime,
            'tickets': o.support_tickets,
            'satisfaction': o.customer_satisfaction
        } for o in ops_data])
        
        # Calculating metrics
        metrics = {
            'sales': {
                'total': float(sales_df['amount'].sum()) if not sales_df.empty else 0,
                'daily_avg': float(sales_df.groupby(sales_df['date'].dt.date)['amount'].sum().mean()) if not sales_df.empty else 0,
                'by_region': sales_df.groupby('region')['amount'].sum().to_dict() if not sales_df.empty else {},
                'by_product': sales_df.groupby('product')['amount'].sum().to_dict() if not sales_df.empty else {}
            },
            'customers': {
                'total': len(churn_df),
                'churned': int(churn_df['churn_flag'].sum()) if not churn_df.empty else 0,
                'churn_rate': float(churn_df['churn_flag'].mean() * 100) if not churn_df.empty else 0,
                'avg_spend': float(churn_df['monthly_spend'].mean()) if not churn_df.empty else 0
            },
            'operations': {
                'avg_uptime': float(ops_df['uptime'].mean()) if not ops_df.empty else 0,
                'total_tickets': int(ops_df['tickets'].sum()) if not ops_df.empty else 0,
                'avg_satisfaction': float(ops_df['satisfaction'].mean()) if not ops_df.empty else 0
            }
        }
        
        return jsonify({
            'success': True,
            'metrics': metrics,
            'period': f'last_{days}_days'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/api/insights/sales', methods=['GET'])
def get_sales_insights():
    """Get AI-generated sales insights"""
    try:
        # Checking if we have recent insights in database
        recent_insight = AIInsights.query.filter_by(
            insight_type='summary',
            metric_type='sales'
        ).order_by(AIInsights.created_at.desc()).first()
        
        if recent_insight and (datetime.now() - recent_insight.created_at).seconds < 3600:
            return jsonify({
                'success': True,
                'insights': json.loads(recent_insight.insight_text),
                'cached': True
            })
        
        # Getting fresh sales data
        sales_data = SalesData.query.filter(
            SalesData.date >= datetime.now() - timedelta(days=90)
        ).all()
        
        sales_df = pd.DataFrame([{
            'date': s.date,
            'sales_amount': s.sales_amount,
            'region': s.region,
            'product_id': s.product_id
        } for s in sales_data])
        
        # Generating insights
        insights = summarizer.analyze_sales_trends(sales_df)
        
        if insights:
            # Cache the insight
            new_insight = AIInsights(
                insight_type='summary',
                metric_type='sales',
                insight_text=json.dumps(insights),
                confidence_score=0.85,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=1)
            )
            db.session.add(new_insight)
            db.session.commit()
        
        return jsonify({
            'success': True,
            'insights': insights,
            'cached': False
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/api/insights/churn', methods=['GET'])
def get_churn_insights():
    """Getting AI-generated churn insights"""
    try:
        churn_data = ChurnData.query.all()
        churn_df = pd.DataFrame([{
            'customer_id': c.customer_id,
            'churn_flag': c.churn_flag,
            'monthly_spend': c.monthly_spend,
            'total_purchases': c.total_purchases,
            'customer_segment': c.customer_segment,
            'signup_date': c.signup_date
        } for c in churn_data])
        
        insights = summarizer.analyze_churn_patterns(churn_df)
        
        return jsonify({
            'success': True,
            'insights': insights
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/api/insights/recommendations', methods=['GET'])
def get_recommendations():
    """Get AI-generated business recommendations"""
    try:
        # Get latest insights
        sales_insights = summarizer.analyze_sales_trends(
            pd.DataFrame([{
                'date': s.date,
                'sales_amount': s.sales_amount
            } for s in SalesData.query.limit(100).all()])
        )
        
        churn_insights = summarizer.analyze_churn_patterns(
            pd.DataFrame([{
                'customer_id': c.customer_id,
                'churn_flag': c.churn_flag,
                'monthly_spend': c.monthly_spend,
                'total_purchases': c.total_purchases,
                'customer_segment': c.customer_segment,
                'signup_date': c.signup_date
            } for c in ChurnData.query.all()])
        )
        
        ops_data = OperationalMetrics.query.limit(30).all()
        ops_insights = {
            'summary': f"Operations metrics over last 30 days",
            'metrics': {
                'avg_uptime': sum(o.server_uptime for o in ops_data) / len(ops_data),
                'avg_satisfaction': sum(o.customer_satisfaction for o in ops_data) / len(ops_data),
                'total_tickets': sum(o.support_tickets for o in ops_data)
            }
        }
        
        recommendations = summarizer.generate_recommendations(
            sales_insights, churn_insights, ops_insights
        )
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/api/forecast/sales', methods=['GET'])
def get_sales_forecast():
    """Get sales forecast"""
    try:
        periods = int(request.args.get('periods', 30))
        
        sales_data = SalesData.query.filter(
            SalesData.date >= datetime.now() - timedelta(days=365)
        ).all()
        
        sales_df = pd.DataFrame([{
            'date': s.date,
            'sales_amount': s.sales_amount
        } for s in sales_data])
        
        # Getting forecast using Prophet
        forecast = sales_predictor.forecast_with_prophet(sales_df, periods)
        
        if not forecast:
            # Fallback to XGBoost
            forecast = sales_predictor.forecast_with_xgboost(sales_df, periods)
        
        return jsonify({
            'success': True,
            'forecast': forecast
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@api_bp.route('/api/data/refresh', methods=['POST'])
def refresh_data():
    """Refreshing sample data (for demo purposes)"""
    try:
        from Database.dp_setup import generate_sample_data
        
        # Clear existing data
        db.session.query(SalesData).delete()
        db.session.query(ChurnData).delete()
        db.session.query(OperationalMetrics).delete()
        db.session.query(AIInsights).delete()
        db.session.commit()
        
        # Generating new sample data
        generate_sample_data()
        
        return jsonify({
            'success': True,
            'message': 'Data refreshed successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500