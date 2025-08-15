# heart-disease-risk-app
This repository contains my Heart disease (CVD) existing risk prediction app, created for my master’s dissertation. The model is built using the Health Survey for England dataset and XGBoost, combining clinical information (such as blood pressure and cholesterol), lifestyle factors (such as smoking, alcohol, and physical activity) and socioeconomic factors (such as martial status, income and housing type), some of these factors are not present in traditonal CVD risk calculators, so I wanted to explore their affect. I’ve added Explainable AI tools (SHAP and LIME) so you can see which factors most influence each prediction. The app also includes a rule-based system that turns these SHAP explanations into simple, personalised lifestyle tips based on NHS guidelines and prior knowledge from my Biomedical Sciences degree. 

In the app, you can explore predictions for a few example patients, see which factors drive their risk up or down using explainable AI, and also view tailored advice based on the SHAP explainations

App Link: https://heart-disease-risk-app-1.streamlit.app/
