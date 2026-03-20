import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime, timedelta
import pandas as pd
import json

class BusinessSummarizer:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
        self.setup_langchain()
        
    def setup_langchain(self):
        """Setup LangChain for orchestration"""
        self.llm = OpenAI(
            temperature=0.3,
            openai_api_key=self.api_key,
            model_name="gpt-3.5-turbo-instruct",
            max_tokens=500
        )
        
        # Summary prompt template
        self.summary_template = PromptTemplate(
            input_variables=["metric_type", "current_value", "previous_value", "context_data"],
            template="""
            You are a business intelligence analyst. Analyze the following {metric_type} data:
            
            Current Value: {current_value}
            Previous Value: {previous_value}
            Context: {context_data}
            
            Please provide:
            1. A plain English summary of what happened (2-3 sentences)
            2. Key factors that contributed to this change
            3. The business impact of this change
            
            Summary:
            """
        )
        
        # Recommendation prompt template
        self.recommendation_template = PromptTemplate(
            input_variables=["insights", "metrics_summary"],
            template="""
            Based on these business insights:
            {insights}
            
            And current metrics:
            {metrics_summary}
            
            Provide 3 actionable recommendations for business leaders:
            1. Short-term actions (next 7 days)
            2. Medium-term strategy (next 30 days)
            3. Long-term initiatives (next 90 days)
            
            For each recommendation, explain the expected impact and implementation steps.
            
            Recommendations:
            """
        )
        
        self.summary_chain = LLMChain(llm=self.llm, prompt=self.summary_template)
        self.recommendation_chain = LLMChain(llm=self.llm, prompt=self.recommendation_template)
    
    def analyze_sales_trends(self, sales_df):
        """Analyzing sales trends and generate insights"""
        try:
            # Calculating key metrics
            current_period = sales_df[sales_df['date'] >= datetime.now() - timedelta(days=7)]
            previous_period = sales_df[
                (sales_df['date'] >= datetime.now() - timedelta(days=14)) & 
                (sales_df['date'] < datetime.now() - timedelta(days=7))
            ]
            
            current_sales = current_period['sales_amount'].sum()
            previous_sales = previous_period['sales_amount'].sum()
            change_percent = ((current_sales - previous_sales) / previous_sales) * 100
            
            # Analyzing by region
            region_performance = current_period.groupby('region')['sales_amount'].sum().to_dict()
            product_performance = current_period.groupby('product_id')['sales_amount'].sum().to_dict()
            
            context = {
                'region_performance': region_performance,
                'product_performance': product_performance,
                'change_percent': change_percent,
                'period': 'last 7 days vs previous 7 days'
            }
            
            # Generating summary using LLM
            summary = self.summary_chain.run(
                metric_type="sales",
                current_value=f"${current_sales:,.2f}",
                previous_value=f"${previous_sales:,.2f}",
                context_data=json.dumps(context, indent=2)
            )
            
            # Getting predictions
            from .predictor import SalesPredictor
            predictor = SalesPredictor()
            forecast = predictor.forecast_with_prophet(sales_df, periods=30)
            
            return {
                'summary': summary,
                'metrics': {
                    'current_sales': current_sales,
                    'previous_sales': previous_sales,
                    'change_percent': change_percent,
                    'region_breakdown': region_performance,
                    'product_breakdown': product_performance
                },
                'forecast': forecast,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Sales analysis error: {e}")
            return None
    
    def analyze_churn_patterns(self, churn_df):
        """Analyzing churn patterns and generate insights"""
        try:
            # Calculating churn metrics
            total_customers = len(churn_df)
            churned_customers = churn_df[churn_df['churn_flag'] == True].shape[0]
            churn_rate = (churned_customers / total_customers) * 100
            
            # Analyzing by segment
            segment_churn = churn_df.groupby('customer_segment')['churn_flag'].mean().to_dict()
            
            # Identifying at-risk customers
            from .predictor import ChurnPredictor
            predictor = ChurnPredictor()
            risk_analysis = predictor.predict_churn_risk(churn_df)
            
            context = {
                'total_customers': total_customers,
                'churned_customers': churned_customers,
                'churn_rate': churn_rate,
                'segment_churn_rates': segment_churn,
                'average_risk': risk_analysis.get('average_risk', 0) if risk_analysis else 0
            }
            
            # Generating summary
            summary = self.summary_chain.run(
                metric_type="customer churn",
                current_value=f"{churn_rate:.1f}%",
                previous_value="N/A",  # Would need historical data
                context_data=json.dumps(context, indent=2)
            )
            
            return {
                'summary': summary,
                'metrics': {
                    'churn_rate': churn_rate,
                    'total_customers': total_customers,
                    'active_customers': total_customers - churned_customers,
                    'segment_churn': segment_churn
                },
                'risk_analysis': risk_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Churn analysis error: {e}")
            return None
    
    def generate_recommendations(self, sales_insights, churn_insights, ops_insights):
        """Generating actionable recommendations based on all insights"""
        try:
            metrics_summary = {
                'sales': {
                    'trend': sales_insights.get('metrics', {}).get('change_percent', 0),
                    'total': sales_insights.get('metrics', {}).get('current_sales', 0)
                },
                'churn': {
                    'rate': churn_insights.get('metrics', {}).get('churn_rate', 0),
                    'at_risk_customers': len(churn_insights.get('risk_analysis', {}).get('customer_risks', []))
                },
                'operations': {
                    'uptime': ops_insights.get('metrics', {}).get('avg_uptime', 0),
                    'satisfaction': ops_insights.get('metrics', {}).get('avg_satisfaction', 0)
                }
            }
            
            insights_text = f"""
            Sales: {sales_insights.get('summary', 'No data')}
            
            Customer Churn: {churn_insights.get('summary', 'No data')}
            
            Operations: {ops_insights.get('summary', 'No data')}
            """
            
            # Generating recommendations
            recommendations = self.recommendation_chain.run(
                insights=insights_text,
                metrics_summary=json.dumps(metrics_summary, indent=2)
            )
            
            # Parse recommendations into structured format
            structured_recommendations = self._parse_recommendations(recommendations)
            
            return {
                'recommendations': structured_recommendations,
                'raw_text': recommendations,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Recommendation generation error: {e}")
            return None
    
    def _parse_recommendations(self, recommendations_text):
        """Parse raw recommendations text into structured format"""
        lines = recommendations_text.strip().split('\n')
        recommendations = []
        current_rec = {}
        
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.')):
                if current_rec:
                    recommendations.append(current_rec)
                current_rec = {'title': line.strip(), 'steps': []}
            elif line.strip().startswith('-') and current_rec:
                current_rec['steps'].append(line.strip()[1:].strip())
            elif line.strip() and current_rec:
                if 'description' not in current_rec:
                    current_rec['description'] = line.strip()
                else:
                    current_rec['description'] += ' ' + line.strip()
        
        if current_rec:
            recommendations.append(current_rec)
            
        return recommendations