import streamlit as st
import pandas as pd
import streamlit_authenticator
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from streamlit_option_menu import option_menu
import streamlit_antd_components as sac
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pathlib import Path
import streamlit_authenticator as stauth
from xgboost import DMatrix
import numpy as np
import os
from datetime import datetime

#set to wide
st.set_page_config(
    page_title="Data Science Studio",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

#topbar
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
st.markdown(f"""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #002B36;">
  <a class="navbar-brand" href="https://instagram.com/danielalvesvm" target="_blank">
    <img src="https://www.logologo.com/img/logologologo.gif" alt="Logo" style="height:40px;"> Data Science Studio
  </a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav mr-auto">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Home <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://instagram.com/danielalvesvm" target="_blank">YouTube</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://instagram.com/danielalvesvm" target="_blank">Twitter</a>
      </li>
    </ul>

  </div>
</nav>
""", unsafe_allow_html=True)

# Function to read and apply CSS styles
def apply_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
apply_css("styles.css")


# Initialize session state for storing models
if 'saved_models' not in st.session_state:
    st.session_state['saved_models'] = []

def save_model(model, name, accuracy, fig, importances):
    with open(f'{name}.pkl', 'wb') as file:
        pickle.dump(model, file)
    st.session_state['saved_models'].append({
        "Name": name,
        "Accuracy": accuracy,
        "Confusion Matrix": fig,
        "Variable Importance": importances
    })
    st.sidebar.write("Model saved successfully!")

def data_manager():
    def list_files(directory='.'):
        """List all CSV files in the specified directory along with their last modified dates and file sizes."""
        files = []
        for filename in os.listdir(directory):
            if filename.endswith('.csv'):
                filepath = os.path.join(directory, filename)
                last_modified = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                size = os.path.getsize(filepath)
                files.append({'Filename': filename, 'Last Modified': last_modified, 'Size (bytes)': size})
        return pd.DataFrame(files)



    with col2:
        st.title("List of Files Uploaded")
        files = list_files()  # You can specify a different directory if needed
        #st.table(files.assign(hack='').set_index('hack'))
        st.write(files)

    with col1:
        st.title("Load New Data")
        option = st.selectbox(
            'Select the csv delimiter',
            (';', ',', '/')
        )

        def save_file(df, filename):
            """Save the DataFrame to the specified filename."""
            df.to_csv(filename, index=False, sep=option)
            return os.path.exists(filename)

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:

            df = pd.read_csv(uploaded_file, delimiter=option)

            if st.button('Save File to Cloud'):
                save_path = uploaded_file.name  # Change this path as needed
                if save_file(df, save_path):
                    st.success(f'File saved successfully at {save_path}')
                else:
                    st.error('An error occurred while saving the file.')




# Function for the Load Data/Run Model page
def load_data_run_model_page():
    st.sidebar.title("Saved Models")
    if st.session_state.get('saved_models'):
        selected_model_name = st.sidebar.selectbox("Select a model to view", options=[model['Name'] for model in st.session_state['saved_models']])
        if selected_model_name:
            selected_model = next((model for model in st.session_state['saved_models'] if model['Name'] == selected_model_name), None)
            if selected_model:
                st.sidebar.write("Model Details:")
                st.sidebar.write(f"Accuracy: {selected_model['Accuracy']}")
                st.sidebar.write("Variable Importance:")
                st.sidebar.write(selected_model['Variable Importance'])
                st.sidebar.write("Confusion Matrix:")
                st.sidebar.plotly_chart(selected_model['Confusion Matrix'])

    # Place the file uploader in the first column
    with col1:
        option = st.selectbox(
            'Select the csv delimiter',
            (';', ',', '/')
        )
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, delimiter=option)
        st.write("Data Summary:")
        st.write(data.head())
        target = st.selectbox("Select the target variable", data.columns)
        explanatory_vars = st.multiselect("Select explanatory variables", data.columns.drop(target))

        if st.button("Run XGBoost"):
            X = data[explanatory_vars]
            y = data[target]
            categorical_vars = X.select_dtypes(include=['object']).columns
            if len(categorical_vars) > 0:
                encoder = OneHotEncoder(sparse_output=False)
                encoded_vars = encoder.fit_transform(X[categorical_vars])
                X = X.drop(categorical_vars, axis=1)
                encoded_var_names = encoder.get_feature_names_out(categorical_vars)
                X = pd.concat([X, pd.DataFrame(encoded_vars, columns=encoded_var_names)], axis=1)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = xgb.XGBClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:,1]

            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, normalize='true')
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            # Creating subplots with dark background
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Confusion Matrix', 'Receiver Operating Characteristic'))

            # Confusion Matrix Plot
            cm_trace = go.Heatmap(z=cm, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'], colorscale='Blues')
            fig.add_trace(cm_trace, 1, 1)

            # Annotations for confusion matrix values
            for i, row in enumerate(cm):
                for j, value in enumerate(row):
                    fig.add_annotation(
                        text=str(round(value, 2)), # Rounded value
                        x=['Predicted 0', 'Predicted 1'][j],
                        y=['Actual 0', 'Actual 1'][i],
                        showarrow=False,
                        font=dict(color="white", size=14), # Font color and size
                        row=1, col=1
                    )

            # ROC Curve Plot
            roc_trace = go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve', line=dict(color='orange'))
            fig.add_trace(roc_trace, 1, 2)
            fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='lightblue'), row=1, col=2)

            # Update layout for dark theme
            fig.update_layout(
                title_text='Model Evaluation Metrics',
                showlegend=True,
                legend=dict(font=dict(color='white')),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white')
            )

            # Update axis labels
            fig.update_xaxes(title_text='Predicted', row=1, col=1, title_font=dict(size=14), tickfont=dict(size=12))
            fig.update_yaxes(title_text='Actual', row=1, col=1, title_font=dict(size=14), tickfont=dict(size=12))
            fig.update_xaxes(title_text='False Positive Rate', row=1, col=2, title_font=dict(size=14), tickfont=dict(size=12))
            fig.update_yaxes(title_text='True Positive Rate', row=1, col=2, title_font=dict(size=14), tickfont=dict(size=12))

            # Displaying Plot
            st.plotly_chart(fig, use_container_width=True)

            feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
            aggregated_importances = {}
            for var in categorical_vars:
                aggregated_importance = feature_importances[feature_importances['feature'].str.startswith(var)].importance.sum()
                aggregated_importances[var] = aggregated_importance
            non_categorical_importances = feature_importances[~feature_importances['feature'].str.startswith(tuple(categorical_vars))]
            total_importances = pd.concat([non_categorical_importances, pd.DataFrame(list(aggregated_importances.items()), columns=['feature', 'importance'])])
            total_importances = total_importances.sort_values('importance', ascending=False)

            st.write("Model Accuracy:", accuracy)
            st.write("Variable Importance", total_importances)

            st.session_state['model'] = model
            st.session_state['accuracy'] = accuracy
            st.session_state['fig'] = fig
            st.session_state['total_importances'] = total_importances

    # Callback for saving the model
    def on_save_button_click():
        model = st.session_state.get('model')
        accuracy = st.session_state.get('accuracy')
        fig = st.session_state.get('fig')
        total_importances = st.session_state.get('total_importances')
        if model:
            custom_name = st.text_input("Enter a name for the model file", value=f"Model_{len(st.session_state['saved_models']) + 1}")
            save_model(model, custom_name, accuracy, fig, total_importances)
            st.write(f"Model {custom_name} saved")
        else:
            st.write("No model to save. Please train a model first.")

    # Save Model Button with Callback
    st.button("Save Model", on_click=on_save_button_click)

# Function for the Predictions page
def predictions_page():
    st.write("Make Predictions")
    predict_data_file = st.file_uploader("Upload a CSV file for predictions", type="csv")

    if predict_data_file is not None:
        predict_data_original = pd.read_csv(predict_data_file, delimiter=';')
        st.write("Prediction Data Summary:")
        st.write(predict_data_original.head())

        if st.button("Predict"):
            model = st.session_state.get('model')
            if model:
                # Process the prediction data
                predict_data = predict_data_original.copy()
                categorical_vars = predict_data.select_dtypes(include=['object']).columns
                if len(categorical_vars) > 0:
                    encoder = OneHotEncoder(sparse_output=False)
                    encoded_vars = encoder.fit_transform(predict_data[categorical_vars])
                    predict_data = predict_data.drop(categorical_vars, axis=1)
                    encoded_var_names = encoder.get_feature_names_out(categorical_vars)
                    predict_data = pd.concat([predict_data, pd.DataFrame(encoded_vars, columns=encoded_var_names)], axis=1)

                # Ensure that prediction data has the same columns as training data
                missing_cols = set(model.get_booster().feature_names) - set(predict_data.columns)
                for col in missing_cols:
                    predict_data[col] = 0
                predict_data = predict_data[model.get_booster().feature_names]

                # Make predictions
                predictions_proba = model.predict_proba(predict_data)[:, 1]  # Assuming binary classification
                predictions = model.predict(predict_data)
                predict_data_original['Probability'] = predictions_proba
                predict_data_original['Prediction'] = predictions

                st.write("Predictions with Original Data:")
                st.write(predict_data_original)

                # Compute Metrics
                true_labels = predict_data_original['Is Closed Won']  # Replace 'True_Label' with your actual label column name
                auc = roc_auc_score(true_labels, predictions_proba)
                accuracy = accuracy_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions)
                conf_matrix = confusion_matrix(true_labels, predictions)

                st.write("AUC:", auc)
                st.write("Accuracy:", accuracy)
                st.write("F1 Score:", f1)
                st.write("Confusion Matrix:")
                st.write(conf_matrix)

                # Find the best cutoff point
                fpr, tpr, thresholds = roc_curve(true_labels, predictions_proba)
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = thresholds[optimal_idx]
                st.write("Best Cutoff Point:", optimal_threshold)

                # Plotting Confusion Matrix
                fig, ax = plt.subplots()
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted Labels')
                ax.set_ylabel('True Labels')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)

            else:
                st.error("No trained model available. Please train a model first.")


#insights or models page
def list_models(root_folder='.'):
    """List all .pkl files in the root folder."""
    return [f for f in os.listdir(root_folder) if f.endswith('1.pkl')]

def load_model(model_path):
    """Load a .pkl model."""
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def get_model_variables(model):
    """Automatically identify variable names and categories from the model."""
    try:
        if hasattr(model, 'feature_names_in_'):
            all_variables = list(model.feature_names_in_)
            categorical_vars = {}
            base_var_names = set()

            for var in all_variables:
                parts = var.split('_')
                if len(parts) > 1:
                    base_var_name = '_'.join(parts[:-1])
                    category = parts[-1]
                    base_var_names.add(base_var_name)

                    if base_var_name not in categorical_vars:
                        categorical_vars[base_var_name] = []
                    categorical_vars[base_var_name].append(category)

            non_categorical_vars = [var for var in all_variables if var.split('_')[0] not in base_var_names]

            return non_categorical_vars, categorical_vars
    except AttributeError:
        pass
    return None, None

def model_page():
    st.title("Model Prediction Page")

    models = list_models()
    selected_model = st.selectbox("Select a model", models)

    if selected_model:
        model = load_model(selected_model)
        non_categorical_vars, categorical_vars = get_model_variables(model)

        if non_categorical_vars is not None or categorical_vars is not None:
            inputs = {}

            # Handle categorical variables with dropdowns
            for cat_var, categories in categorical_vars.items():
                selected_category = st.selectbox(cat_var, options=categories)
                # Set the selected category to 1 and others to 0
                for category in categories:
                    inputs[f"{cat_var}_{category}"] = 1 if category == selected_category else 0

            # Handle non-categorical variables
            for var in non_categorical_vars:
                inputs[var] = st.number_input(f"{var}", format="%f")

            if st.button("Predict"):
                # Debugging: Display the input values
                st.write("Inputs:", inputs)

                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba([list(inputs.values())])
                    # Debugging: Display raw probabilities
                    st.write("Raw probabilities:", probabilities)

                    prediction_probability = probabilities[0][1]  # Adjust index if needed
                else:
                    prediction = model.predict([list(inputs.values())])
                    prediction_probability = prediction[0]

                prediction_percentage = prediction_probability * 100
                formatted_prediction = "{:.2f}%".format(prediction_percentage)

                st.write("Prediction Probability:", formatted_prediction)
        else:
            st.write("Could not determine the variables used in the model.")


def main():

    #Logout Button
    st.sidebar.markdown(f"<p style='display: inline-block; margin-right: 10px;'>Hello, {name}!</p>",
                        unsafe_allow_html=True)
    authenticator.logout("Logout", "sidebar")

    # Display title with icon in the sidebar
    st.sidebar.header(" ")

    # Sidebar
    with st.sidebar:

        selected = sac.menu([
            sac.MenuItem('Load Data', icon='file-arrow-up-fill'),
            sac.MenuItem('Data Management', icon='file-arrow-up-fill'),
            sac.MenuItem('Models', icon='box-fill', children=[
                sac.MenuItem('XGBoost', icon='robot', tag=sac.Tag('NEW', color='green', bordered=False)),
                sac.MenuItem('Neural Network', icon='postage', children=[
                    sac.MenuItem('RNN', icon='snow3'),
                    sac.MenuItem('LSTM', icon='snow3'),
                    sac.MenuItem('Deep Learning' * 1, icon='snow3'),
                ]),
            ]),
            sac.MenuItem('Predictions', icon='magic'),
            sac.MenuItem('Insights', icon='apple'),
            sac.MenuItem(type='divider'),
            sac.MenuItem('Contact', type='group', children=[
                sac.MenuItem('@danielalvesvm', icon='instagram', href='https://instagram.com/danielalvesvm'),
                sac.MenuItem('danielalvesvm@gmail.com', icon='envelope', href='mailto:danielalvesvm@gmail.com'),
            ]),
        ], indent=19, open_index=[1, 3], open_all=True)
        #design at https://ant.design/components/menu#menu and icons https://icons.getbootstrap.com/

    # Execute the corresponding function based on the selection
    if selected == "Load Data":
        load_data_run_model_page()
    elif selected == "Predictions":
        predictions_page()
    elif selected == "Insights":
        model_page()
    elif selected == "Data Management":
        data_manager()



#authfirst
names = ["Daniel Alves", "John Alves", "Thiago Roberto"]
usernames = ["dalves", "jalves", "thiago"]
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(
    names,
    usernames,
    hashed_passwords,
    "cookie_Pws",
    "abcdef",
    cookie_expiry_days=30
)

col1, col2, col3 = st.columns(3)
with col2:

    name, authentication_status, username = authenticator.login("Login", "main")

    if authentication_status == False:
        st.error("Username/Password is incorrect")

    if authentication_status == None:
        st.warning("Please enter your Username and Password")

if authentication_status:
    if __name__ == "__main__":
        main()