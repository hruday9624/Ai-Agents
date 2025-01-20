from crewai import Crew,Process
from tasks import (data_integration_task,research_task,model_training_task,model_evaluation_task,
                   visualization_task,senior_data_scientist_task,client_communication_task)
from agents import (senior_data_scientist,stock_market_analyst,financial_operations_coordinator,
                    model_trainer_agent,model_evaluator_agent,visualization_agent,data_operations_specialist)
import os
# Access the Alpha Vantage API key from the environment
alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

crew = Crew(
    agents=[
        data_operations_specialist,
        stock_market_analyst,
        model_trainer_agent,
        model_evaluator_agent,
        visualization_agent,
        senior_data_scientist,
        financial_operations_coordinator
    ],
    tasks=[
        data_integration_task,
        research_task,
        model_training_task,
        model_evaluation_task,
        visualization_task,
        senior_data_scientist_task,
        client_communication_task
    ],
    verbose=True
)

inputs = {
    "asset": "US WTI Crude Oil (USOIL)",  # Asset to forecast
    "intervals": ["weekly"],  # Forecasting intervals
    "historical_data_source": "Alpha Vantage",  # Using Alpha Vantage API for historical data
    "alpha_vantage_api_key": alpha_vantage_api_key,  # Passing API key to input configuration
    "data_requirements": {
        "years_of_historical_data": 5,
        "short_window_for_predictions": 100
    },
    "model": {
        "type": "LSTM",
        "accuracy_goal": 85,
        "evaluation_metrics": ["MAE", "RMSE", "R2"]
    },
    "daily_tasks": {
        "plot_comparison": "Plot actual vs predicted prices daily",
        "save_predictions": "Store daily predicted prices in a file for iterative plots"
    },
    "timeline": {
        "delivery_days": 10,
        "priority": "High-quality over rushed output"
    },
    "visualization": "Interactive charts to compare actual and predicted prices",
    "quote_discussion": "Discuss cost and effort for delivering this solution",
    "deployment_notes": {
        "training_frequency": "Train once on historical data",
        "inference_frequency": "Perform daily predictions with sliding window"
    }
}


# Kick off the crew
result = crew.kickoff(inputs=inputs)
print(result)