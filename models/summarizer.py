import openai
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from datetime import datetime, timedelta
import pandas as pd
import json
import sys
import os

# Adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BusinessSummarizer:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
        self.setup_langchain()
        
    def setup_langchain(self):
        """Setup LangChain for orchestration"""
        try:
            self.llm = OpenAI(
                temperature=0.3,
                openai_api_key=self.api_key,
                model="gpt-3.5-turbo-instruct",
                max_tokens=500
            )
        except Exception as e:
            print(f"Error setting up LangChain: {e}")
            # Fallback to direct OpenAI calls
            self.llm = None
        
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
        
        if self.llm:
            self.summary_chain = LLMChain(llm=self.llm, prompt=self.summary_template)
            self.recommendation_chain = LLMChain(llm=self.llm, prompt=self.recommendation_template)
    
    def _call_openai_direct(self, prompt):
        """Fallback method to call OpenAI directly"""
        try:
            response = openai.Completion.create(
                engine="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"OpenAI direct call error: {e}")
            return "Unable to generate insights at this time."
    
    def analyze_sales_trends(self, sales_df):
        """Analyze sales trends and generate insights"""
        try:
            # Calculating key metrics
            current_period = sales_df[sales_df['date'] >= datetime.now() - timedelta(days=7)]
            previous_period = sales_df[
                (sales_df['date'] >= datetime.now() - timedelta(days=14)) & 
                (sales_df['date'] < datetime.now() - timedelta(days=7))
            ]
            
            current_sales = current_period['sales_amount'].sum() if not current_period.empty else 0
            previous_sales = previous_period['sales_amount'].sum() if not previous_period.empty else 0
            
            if previous_sales > 0:
                change_percent = ((current_sales - previous_sales) / previous_sales) * 100
            else:
                change_percent = 0
            
            # Analyzing by region
            region_performance = current_period.groupby('region')['sales_amount'].sum().to_dict() if not current_period.empty else {}
            product_performance = current_period.groupby('product_id')['sales_amount'].sum().to_dict() if not current_period.empty else {}
            
            context = {
                'region_performance': region_performance,
                'product_performance': product_performance,
                'change_percent': change_percent,
                'period': 'last 7 days vs previous 7 days'
            }
            
            # Generating summary using LLM
            if self.llm and hasattr(self, 'summary_chain'):
                try:
                    summary = self.summary_chain.run(
                        metric_type="sales",
                        current_value=f"${current_sales:,.2f}",
                        previous_value=f"${previous_sales:,.2f}",
                        context_data=json.dumps(context, indent=2)
                    )
                except:
                    summary = self._generate_fallback_sales_summary(current_sales, previous_sales, change_percent)
            else:
                summary = self._generate_fallback_sales_summary(current_sales, previous_sales, change_percent)
            
            # Getting predictions
            try:
                from .predictor import SalesPredictor
                predictor = SalesPredictor()
                forecast = predictor.forecast_with_prophet(sales_df, periods=30)
            except Exception as e:
                print(f"Forecast error: {e}")
                forecast = None
            
            return {
                'summary': summary,
                'metrics': {
                    'current_sales': float(current_sales),
                    'previous_sales': float(previous_sales),
                    'change_percent': float(change_percent),
                    'region_breakdown': region_performance,
                    'product_breakdown': product_performance
                },
                'forecast': forecast,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Sales analysis error: {e}")
            return {
                'summary': "Sales analysis temporarily unavailable. Please try again later.",
                'metrics': {},
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_fallback_sales_summary(self, current_sales, previous_sales, change_percent):
        """Generating fallback summary without LLM"""
        if change_percent > 0:
            trend = f"increased by {change_percent:.1f}%"
            impact = "positive growth"
        elif change_percent < 0:
            trend = f"decreased by {abs(change_percent):.1f}%"
            impact = "requires attention"
        else:
            trend = "remained stable"
            impact = "steady performance"
        
        return f"Sales {trend} from ${previous_sales:,.2f} to ${current_sales:,.2f}. This represents {impact} in the current period. Further analysis is needed to identify specific drivers of this change."
    
    def analyze_churn_patterns(self, churn_df):
        """Analyzing churn patterns and generating insights"""
        try:
            # Calculating churn metrics
            total_customers = len(churn_df)
            churned_customers = churn_df[churn_df['churn_flag'] == True].shape[0] if 'churn_flag' in churn_df.columns else 0
            churn_rate = (churned_customers / total_customers) * 100 if total_customers > 0 else 0
            
            # Analyzing by segment
            if 'customer_segment' in churn_df.columns and 'churn_flag' in churn_df.columns:
                segment_churn = churn_df.groupby('customer_segment')['churn_flag'].mean().to_dict()
            else:
                segment_churn = {}
            
            # Identify at-risk customers
            try:
                from .predictor import ChurnPredictor
                predictor = ChurnPredictor()
                risk_analysis = predictor.predict_churn_risk(churn_df)
            except Exception as e:
                print(f"Risk analysis error: {e}")
                risk_analysis = {'average_risk': 0.5}
            
            context = {
                'total_customers': total_customers,
                'churned_customers': churned_customers,
                'churn_rate': churn_rate,
                'segment_churn_rates': segment_churn,
                'average_risk': risk_analysis.get('average_risk', 0) if risk_analysis else 0
            }
            
            # Generating summary
            summary = f"Customer churn rate is currently {churn_rate:.1f}% with {churned_customers} customers churned out of {total_customers} total. "
            
            if segment_churn:
                highest_churn_segment = max(segment_churn, key=segment_churn.get)
                summary += f"The {highest_churn_segment} segment shows the highest churn rate at {segment_churn[highest_churn_segment]*100:.1f}%. "
            
            summary += f"Average churn risk across all customers is {risk_analysis.get('average_risk', 0)*100:.1f}%."
            
            return {
                'summary': summary,
                'metrics': {
                    'churn_rate': float(churn_rate),
                    'total_customers': int(total_customers),
                    'active_customers': int(total_customers - churned_customers),
                    'segment_churn': {k: float(v) for k, v in segment_churn.items()}
                },
                'risk_analysis': risk_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Churn analysis error: {e}")
            return {
                'summary': "Churn analysis temporarily unavailable.",
                'metrics': {},
                'timestamp': datetime.now().isoformat()
            }
    
    def generate_recommendations(self, sales_insights, churn_insights, ops_insights):
        """Generate actionable recommendations based on all insights"""
        try:
            metrics_summary = {
                'sales': {
                    'trend': sales_insights.get('metrics', {}).get('change_percent', 0) if sales_insights else 0,
                    'total': sales_insights.get('metrics', {}).get('current_sales', 0) if sales_insights else 0
                },
                'churn': {
                    'rate': churn_insights.get('metrics', {}).get('churn_rate', 0) if churn_insights else 0,
                    'at_risk_customers': len(churn_insights.get('risk_analysis', {}).get('customer_risks', [])) if churn_insights else 0
                },
                'operations': {
                    'uptime': ops_insights.get('metrics', {}).get('avg_uptime', 0) if ops_insights else 0,
                    'satisfaction': ops_insights.get('metrics', {}).get('avg_satisfaction', 0) if ops_insights else 0
                }
            }
            
            # Generating structured recommendations without LLM
            recommendations = [
                {
                    'title': 'Short-term: Optimize Customer Retention',
                    'description': 'Target high-risk customers with personalized retention campaigns',
                    'steps': ['Identify top 100 high-risk customers', 'Create personalized discount offers', 'Schedule check-in calls']
                },
                {
                    'title': 'Medium-term: Enhance Sales Performance',
                    'description': 'Focus on high-performing regions and products',
                    'steps': ['Analyze best-performing regions', 'Reallocate marketing budget', 'Expand successful product lines']
                },
                {
                    'title': 'Long-term: Improve Operational Excellence',
                    'description': 'Invest in automation and customer experience',
                    'steps': ['Implement AI chatbots', 'Optimize support workflows', 'Launch customer success program']
                }
            ]
            
            return {
                'recommendations': recommendations,
                'raw_text': "Based on current data, focus on customer retention, sales optimization, and operational improvements.",
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Recommendation generation error: {e}")
            return {
                'recommendations': [],
                'timestamp': datetime.now().isoformat()
            }
    
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