import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from tensorflow import keras
from tensorflow.keras import layers
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Loan Approval Dashboard", layout="wide")

st.title("Loan Approval Prediction Dashboard")
st.markdown("Compare Fuzzy Logic vs Neural Network predictions")

# Sidebar for file upload
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Data preprocessing
    df = df.drop(columns=['name', 'city'])
    df['loan_approved'] = df['loan_approved'].astype(int)
    
    features = ['income', 'credit_score', 'loan_amount', 'years_employed']
    X = df[features].values
    y = df['loan_approved'].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train models
    with st.spinner("Training models..."):
        # Fuzzy Logic System
        universe = np.arange(0, 1.01, 0.01)
        
        income = ctrl.Antecedent(universe, 'income')
        credit = ctrl.Antecedent(universe, 'credit')
        loan = ctrl.Antecedent(universe, 'loan')
        experience = ctrl.Antecedent(universe, 'experience')
        approval = ctrl.Consequent(universe, 'approval')
        
        for var in [income, credit, loan, experience]:
            var['low'] = fuzz.trimf(universe, [0, 0, 0.4])
            var['medium'] = fuzz.trimf(universe, [0.3, 0.5, 0.7])
            var['high'] = fuzz.trimf(universe, [0.6, 1, 1])
        
        approval['reject'] = fuzz.trimf(universe, [0, 0, 0.5])
        approval['approve'] = fuzz.trimf(universe, [0.5, 1, 1])
        
        rules = [
            ctrl.Rule(credit['high'] & income['high'], approval['approve']),
            ctrl.Rule(credit['high'] & experience['high'], approval['approve']),
            ctrl.Rule(credit['medium'] & income['high'] & experience['medium'], approval['approve']),
            ctrl.Rule(credit['medium'] & income['medium'] & experience['medium'], approval['approve']),
            ctrl.Rule(credit['high'] & loan['medium'], approval['approve']),
            ctrl.Rule(credit['low'], approval['reject']),
            ctrl.Rule(loan['high'], approval['reject']),
            ctrl.Rule(income['low'] & credit['medium'], approval['reject']),
            ctrl.Rule(experience['low'] & loan['high'], approval['reject']),
            ctrl.Rule(credit['medium'] & loan['high'], approval['reject']),
            ctrl.Rule(income['medium'] & loan['high'], approval['reject'])
        ]
        
        system = ctrl.ControlSystem(rules)
        
        fuzzy_preds = []
        for row in X_test:
            sim = ctrl.ControlSystemSimulation(system)
            try:
                sim.input['income'] = row[0]
                sim.input['credit'] = row[1]
                sim.input['loan'] = row[2]
                sim.input['experience'] = row[3]
                sim.compute()
                score = sim.output['approval']
                fuzzy_preds.append(1 if score > 0.5 else 0)
            except:
                fuzzy_preds.append(0)
        
        # Neural Network
        model = keras.Sequential([
            keras.Input(shape=(X_train.shape[1],)),
            layers.Dense(8, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
        nn_preds = (model.predict(X_test) > 0.5).astype(int).flatten()
    
    # Calculate accuracies
    fuzzy_acc = accuracy_score(y_test, fuzzy_preds)
    nn_acc = accuracy_score(y_test, nn_preds)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Samples", len(y_test))
    with col2:
        st.metric("Fuzzy Logic Accuracy", f"{fuzzy_acc:.2%}")
    with col3:
        st.metric("Neural Network Accuracy", f"{nn_acc:.2%}")
    
    # Comparison chart
    st.subheader("Model Performance Comparison")
    comparison_df = pd.DataFrame({
        'Model': ['Fuzzy Logic', 'Neural Network'],
        'Accuracy': [fuzzy_acc, nn_acc]
    })
    
    fig = px.bar(comparison_df, x='Model', y='Accuracy', 
                 color='Model', text=comparison_df['Accuracy'].apply(lambda x: f'{x:.2%}'),
                 title='Accuracy Comparison')
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis_tickformat='.0%', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Predictions comparison
    st.subheader("Predictions Comparison")
    
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Fuzzy Logic': fuzzy_preds,
        'Neural Network': nn_preds
    })
    
    # Show first 20 predictions
    st.dataframe(results_df.head(20), use_container_width=True)
    
    # Confusion matrix visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Fuzzy Logic Confusion Matrix**")
        fuzzy_cm = pd.crosstab(pd.Series(y_test, name='Actual'), 
                               pd.Series(fuzzy_preds, name='Predicted'))
        fig_fuzzy = px.imshow(fuzzy_cm, text_auto=True, 
                               title="Fuzzy Logic",
                               labels=dict(x="Predicted", y="Actual", color="Count"))
        st.plotly_chart(fig_fuzzy, use_container_width=True)
    
    with col2:
        st.markdown("**Neural Network Confusion Matrix**")
        nn_cm = pd.crosstab(pd.Series(y_test, name='Actual'), 
                           pd.Series(nn_preds, name='Predicted'))
        fig_nn = px.imshow(nn_cm, text_auto=True,
                           title="Neural Network",
                           labels=dict(x="Predicted", y="Actual", color="Count"))
        st.plotly_chart(fig_nn, use_container_width=True)
    
    # Feature distribution
    st.subheader("Feature Distributions")
    
    df_display = df.copy()
    fig = px.box(df_display[features], title="Feature Distributions by Loan Status",
                 color=df_display['loan_approved'].map({0: 'Rejected', 1: 'Approved'}))
    st.plotly_chart(fig, use_container_width=True)
    
    # Individual prediction tester
    st.subheader("Test Individual Prediction")
    st.markdown("Adjust the values below to see how each model predicts")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        test_income = st.slider("Income", 0.0, 1.0, 0.5)
        test_credit = st.slider("Credit Score", 0.0, 1.0, 0.5)
        test_loan = st.slider("Loan Amount", 0.0, 1.0, 0.5)
        test_experience = st.slider("Years Employed", 0.0, 1.0, 0.5)
    
    if st.button("Predict"):
        # Fuzzy prediction
        sim = ctrl.ControlSystemSimulation(system)
        try:
            sim.input['income'] = test_income
            sim.input['credit'] = test_credit
            sim.input['loan'] = test_loan
            sim.input['experience'] = test_experience
            sim.compute()
            fuzzy_score = sim.output['approval']
            fuzzy_result = "Approved" if fuzzy_score > 0.5 else "Rejected"
        except:
            fuzzy_score = 0
            fuzzy_result = "Error"
        
        # NN prediction (using the scaled input for demo)
        test_input = scaler.transform([[test_income, test_credit, test_loan, test_experience]])
        nn_score = model.predict(test_input, verbose=0)[0][0]
        nn_result = "Approved" if nn_score > 0.5 else "Rejected"
        
        with col2:
            st.markdown("### Results")
            st.info(f"**Fuzzy Logic:** {fuzzy_result} (Score: {fuzzy_score:.3f})")
            st.info(f"**Neural Network:** {nn_result} (Score: {nn_score:.3f})")

else:
    # Instructions when no file is uploaded
    st.info("👈 Please upload a CSV file to get started")
    
    st.markdown("""
    ### Expected CSV format:
    Your CSV should contain the following columns:
    - `name` (will be dropped)
    - `city` (will be dropped)
    - `income` (numeric)
    - `credit_score` (numeric)
    - `loan_amount` (numeric)
    - `years_employed` (numeric)
    - `loan_approved` (0 or 1)
    
    ### Features:
    - **Model Comparison**: Compare Fuzzy Logic vs Neural Network accuracy
    - **Visual Analytics**: View distributions and confusion matrices
    - **Interactive Testing**: Test individual predictions with sliders
    - **Real-time Results**: See predictions on test data
    """)
    
    # Sample data generator
    if st.button("Load Sample Data"):
        np.random.seed(42)
        n_samples = 100
        sample_df = pd.DataFrame({
            'name': [f'Person_{i}' for i in range(n_samples)],
            'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n_samples),
            'income': np.random.uniform(30000, 150000, n_samples),
            'credit_score': np.random.uniform(300, 850, n_samples),
            'loan_amount': np.random.uniform(5000, 500000, n_samples),
            'years_employed': np.random.uniform(0, 30, n_samples),
            'loan_approved': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
        })
        sample_df.to_csv('sample_loan_data.csv', index=False)
        st.success("Sample data created as 'sample_loan_data.csv'")
        st.markdown("Please upload this file using the sidebar")