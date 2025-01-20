from crewai import Task
from agents import (senior_data_scientist,stock_market_analyst,financial_operations_coordinator,
                    model_trainer_agent,model_evaluator_agent,visualization_agent,data_operations_specialist)
#from tools import tool

# Task for Data Operations Specialist (Data Integrator)
data_integration_task = Task(
    description=(
        "Integrate and preprocess stock market data using Python libraries (e.g., yfinance, Alpha Vantage, pandas). "
        "Input details such as stock symbols, time frames, and other parameters will be dynamically provided. "
        "Ensure the data is cleaned (handling missing values, outliers, and inconsistencies), formatted consistently, "
        "and saved as a CSV file for further processing by subsequent agents."
    ),
    expected_output=(
        "A preprocessed dataset containing clean, complete, and consistent stock market data in CSV format, "
        "ready for further analysis."
    ),
    agent=data_operations_specialist,
    output_file="preprocessed_data.csv",  # Final cleaned dataset
    forward_data=True  # Ensure data is passed to the next agent in the workflow
)


# Task for Stock Market Analyst (Research Specialist)
research_task = Task(
    description=(
        "Analyze stock market trends, patterns, and anomalies using historical data and indicators. "
        "Generate visualizations and actionable insights for forecasting. Save the executed code for reproducibility."
    ),
    expected_output=(
        "1. A structured Markdown report with identified trends, anomalies, and insights. "
        "2. Visualizations (plots saved as PNG files). "
        "3. The Python script used for analysis saved as a .py file."
    ),
    agent=stock_market_analyst,
    output_file="market_trends_analysis.md",
    additional_outputs={
        "visualizations_folder": "visualizations/",
        "code_file": "analysis_script.py"
    },
    forward_output=True
)


# Task for Model Trainer Agent
model_training_task = Task(
    description=(
        "Train, fine-tune, and evaluate forecasting models using preprocessed stock market data and insights provided by the analyst. "
        "Optimize models for performance and save training logs, evaluation metrics, visualizations, and the training code."
    ),
    expected_output=(
        "1. Serialized forecasting models (e.g., ARIMA.pkl, LSTM.h5). "
        "2. Training logs detailing the process and hyperparameters. "
        "3. Evaluation metrics (RMSE, MAPE, R-squared). "
        "4. Visualizations of model performance (e.g., predicted vs. actual plots). "
        "5. Code files used to train the models."
    ),
    agent=model_trainer_agent,
    output_file="training_logs.md",  # Markdown report summarizing logs and metrics
    additional_outputs={
        "models_folder": "trained_models/",  # Save trained models here
        "metrics_file": "evaluation_metrics.csv",  # Metrics saved as CSV
        "visualizations_folder": "model_performance_plots/",  # Performance plots
        "code_file": "training_code.py"  # Script used for model training
    },
    dependencies=[data_integration_task, research_task],  # Dependencies for input
    forward_output=True  # Share outputs with subsequent tasks
)


# Task for Model Evaluator Agent
model_evaluation_task = Task(
    description=(
        "Evaluate the performance of trained models using metrics like RMSE, MAE, and MAPE. "
        "Provide a detailed report with the evaluation results, including strengths, weaknesses, and deployment recommendations."
    ),
    expected_output=(
        "1. Evaluation metrics: RMSE, MAE, MAPE for each trained model. "
        "2. Evaluation report summarizing model performance and insights. "
        "3. Recommendations for model deployment based on performance."
    ),
    agent=model_evaluator_agent,
    output_file="model_evaluation_report.md",  # Markdown report containing evaluation details
    additional_outputs={
        "evaluation_metrics": "evaluation_metrics.csv",  # Save metrics in CSV format
        "evaluation_code_file": "evaluation_code.py"  # Save evaluation code used for the task
    },
    dependencies=[model_training_task],  # Model training must complete before evaluation
    forward_output=True  # Forward the outputs to the next agent or task
)


# Task for Visualization Agent
visualization_task = Task(
    description=(
        "Create visualizations that clearly present model results, trends, and insights. "
        "These visualizations will include time series plots, trend analysis charts, and performance comparison plots. "
        "The goal is to help stakeholders understand the model's performance and trends."
    ),
    expected_output=(
        "1. Time series plots displaying actual vs. predicted values. "
        "2. Trend analysis charts for insights into stock market behavior. "
        "3. Performance comparison plots between different models. "
        "4. A comprehensive visualization report."
    ),
    agent=visualization_agent,
    output_file="visualization_results.md",  # Markdown report that links or embeds the visualizations
    additional_outputs={
        "visualizations_folder": "model_visualizations/",  # Save plots in this folder
        "visualization_code_file": "visualization_code.py"  # Save the code used for generating visualizations
    },
    dependencies=[model_evaluation_task],  # Visualization depends on model evaluation completion
    forward_output=True  # Forward visualizations and code to the next agent/task
)


# Task for Senior Data Scientist (Lead Agent)
senior_data_scientist_task = Task(
    description=(
        "Oversee and guide the entire forecasting pipeline, ensuring all outputs meet quality standards and align with client requirements. "
        "Compile and validate models, visualizations, and code from all previous agents into a final deliverable."
    ),
    expected_output=(
        "1. Final report containing actionable insights, validated models, and a summary of findings. "
        "2. A compiled Python file containing all code generated by previous agents for reproducibility."
    ),
    agent=senior_data_scientist,
    output_file="final_client_report.md",  # Final report summarizing the entire process
    additional_outputs={
        "compiled_code_file": "compiled_forecasting_pipeline.py"  # Compile all code from previous agents into one file
    },
    dependencies=[visualization_task, model_evaluation_task, model_training_task, research_task, data_integration_task],  # All prior tasks must complete
    forward_output=True  # Forward the final report and compiled code to the next agent or storage
)


# Task for Financial Operations Coordinator (Client Liaison)
client_communication_task = Task(
    description=(
        "Communicate the results, visualizations, and insights to the client. Ensure the report is tailored to their business objectives, "
        "providing clear, actionable recommendations based on data analysis and model outcomes."
    ),
    expected_output=(
        "1. A tailored client-facing report that includes insights, visualizations, and recommendations aligned with the client’s business objectives. "
        "2. Clear, actionable takeaways from the data science and forecasting pipeline."
    ),
    agent=financial_operations_coordinator,
    output_file="client_presentation.md",  # Client presentation or summary report
    additional_outputs={
        "interactive_visualizations": "interactive_visualizations.html",  # Optional: HTML file with interactive plots
        "client_summary_file": "client_summary.txt"  # Optional: Text summary for a quick overview
    },
    dependencies=[senior_data_scientist_task, visualization_task],  # The task depends on prior outputs (final report, visualizations)
    forward_output=True  # Forward the final presentation to the next step or deliverable storage
)
