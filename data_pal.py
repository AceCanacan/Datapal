import openai
import streamlit as st
import pandas as pd

def interpret_csv(data):
    """Interpret the dataset and its columns."""
    
    # Check if interpretations have already been generated
    if "all_interpretations" in st.session_state:
        return st.session_state.all_interpretations

    # Initialize an empty string to store all interpretations
    all_interpretations = ""
    
    # Get an overview of the dataset by reading its head
    dataset_head = data.head().to_string()

    # Ask the model to interpret the dataset as a whole
    dataset_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {"role": "system", "content": "Interpret the following dataset overview."},
            {"role": "user", "content": dataset_head}
        ]
    )
    dataset_interpretation = dataset_response.choices[0].message['content'].strip()
    all_interpretations += f"Dataset Overview:\n{dataset_interpretation}\n\n"

    # Interpret each variable/column individually
    for column in data.columns:
        sample_data = data[column].head(5).to_string()
        column_response = openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system", "content": "Interpret the following column from a dataset, considering the sample data."},
                {"role": "user", "content": f"Column: {column}\nSample Data:\n{sample_data}"}
            ]
        )
        column_interpretation = column_response.choices[0].message['content'].strip()
        all_interpretations += f"Column '{column}':\n{column_interpretation}\n\n"

    # Update the session state with the new interpretations
    st.session_state.all_interpretations = all_interpretations

    return all_interpretations

# # # # # # # # # # # # # # 

def manage_interpretations(prompt):
    """Manage the editing of interpretations."""
    # Use the model to generate the new interpretation
    new_interpretation_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": f"Edit the following dataset interpretation based on the user's input:\n{st.session_state.all_interpretations}. But do not overrite and delete everything when they suggest"
            },
            {"role": "user", "content": prompt}
        ]
    )
    new_interpretation = new_interpretation_response.choices[0].message['content'].strip()
    
    # Update the all_interpretations string
    st.session_state.all_interpretations = new_interpretation

    # Prompt the model to generate a brief description of what was edited
    description_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": "Provide a brief description of what was just edited in the dataset interpretation."
            },
            {"role": "assistant", "content": f"The dataset interpretation has been updated to:\n{new_interpretation}"}
        ]
    )
    brief_description = description_response.choices[0].message['content'].strip()

    # Display the brief description to the user
    with st.chat_message("assistant"):
        st.markdown(f"Edit completed. {brief_description}")

# # # # # # # # # # # # # # 
# # # # # # # # # # # # # # 
# # # # # # # # # # # # # # 
# # # # # # # # # # # # # # 

def generate_visualization_questions(column_interpretation, all_interpretations):
    """Generate questions and Python code for visualizations based on dataset interpretations."""
    
    # Check if visualizations have already been generated
    if st.session_state.all_visualizations:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.all_visualizations)
        return

    # Format the user prompt for better readability
    user_prompt = f"Given the following dataset information:\n" \
                  f"Dataset Interpretation:\n{all_interpretations}\n" \
                  f"Column Interpretations:\n{column_interpretation}\n" \
                  "What potential visualizations can you come up with? " \
                  "Make sure to generate code for each type of questions which are 'Distribution', 'Relationship', 'Composition', 'Comparison'. " \
                  "Make sure that for each visualization, there is a block of Python code that will generate that visualization. " \
                  "Print them in Python block codes."
    
    # Use the model to generate Python code and questions for the suggested visualization
    visualization_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {"role": "system", "content": "You are a data science assistant specialized in generating visualizations."},
            {"role": "user", "content": user_prompt}
        ]
    )
    visualization_output = visualization_response.choices[0].message['content'].strip()
    
    # Save the generated visualizations to session state
    st.session_state.all_visualizations = visualization_output

    # Display the Python code and the questions to the user
    with st.chat_message("assistant"):
        st.markdown(visualization_output)

# # # # # # # # # # # # # # 

def manage_visualizations(prompt):
    """Manage the editing of visualizations."""

    if 'all_visualizations' not in st.session_state:
        st.session_state.all_visualizations = ""

    # Use the model to generate the new visualizations
    new_visualization_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": f"Edit the following visualizations based on the user's input:\n{st.session_state.all_visualizations} Retain the existing output, do not delete them. Just add the edits"
            },
            {"role": "user", "content": prompt}
        ]
    )
    new_visualization = new_visualization_response.choices[0].message['content'].strip()
    
    # Update the all_visualizations string in session state
    st.session_state.all_visualizations = new_visualization

    # Prompt the model to generate a brief description of what was edited
    description_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": "Provide a brief description of what was just edited in the visualizations."
            },
            {"role": "assistant", "content": f"The visualizations have been updated to:\n{new_visualization}"}
        ]
    )
    brief_description = description_response.choices[0].message['content'].strip()

    # Display the brief description to the user
    with st.chat_message("assistant"):
        st.markdown(f"Edit completed. {brief_description}")

# # # # # # # # # # # # # # 

def brainstorm_visualizations(prompt, all_interpretations):
    """Help the user brainstorm possible visualizations based on the dataset."""
    
    # Create a detailed prompt for the model
    brainstorm_prompt = f"You are a data science assistant. The user is uncertain about what visualizations to create. " \
                        f"Based on the following dataset interpretations, suggest a possible visualization:\n{all_interpretations}"
    
    # Use the model to generate suggestions for visualizations
    brainstorm_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {"role": "system", "content": brainstorm_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    brainstorm_output = brainstorm_response.choices[0].message['content'].strip()
    
    # Display the suggestions to the user
    with st.chat_message("assistant"):
        st.markdown(brainstorm_output)
        
# # # # # # # # # # # # # # 
# # # # # # # # # # # # # # 
# # # # # # # # # # # # # # 
# # # # # # # # # # # # # #

def generate_data_cleaning_edits(column_interpretation, all_interpretations):
    """Generate Python code for data cleaning based on dataset interpretations."""

    # Check if data cleaning has already been generated
    if st.session_state.all_data_cleaning:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.all_data_cleaning)
        return

    user_prompt = f"Given the following dataset information:\n" \
                  f"Dataset Interpretation:\n{all_interpretations}\n" \
                  f"Column Interpretations:\n{column_interpretation}\n" \
                  "What potential data cleaning issues might arise and how would you solve them? " \
                  "Generate Python code for each type of data cleaning. " \
                  "Print them in Python block codes."

    data_cleaning_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {"role": "system", "content": "You are a data science assistant specialized in data cleaning."},
            {"role": "user", "content": user_prompt}
        ]
    )
    data_cleaning_output = data_cleaning_response.choices[0].message['content'].strip()

    # Save the generated data cleaning to session state
    st.session_state.all_data_cleaning = data_cleaning_output

    with st.chat_message("assistant"):
        st.markdown(data_cleaning_output)

# # # # # # # # # # # # # # 

def manage_data_cleaning_edits(prompt):
    """Manage the editing of data cleaning."""

    if 'all_data_cleaning' not in st.session_state:
        st.session_state.all_data_cleaning = ""

    new_data_cleaning_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": f"Edit the following data cleaning based on the user's input:\n{st.session_state.all_data_cleaning}. Retain the existing output, do not delete them. Just add the edits"
            },
            {"role": "user", "content": prompt}
        ]
    )
    new_data_cleaning = new_data_cleaning_response.choices[0].message['content'].strip()

    st.session_state.all_data_cleaning = new_data_cleaning

    description_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": "Provide a brief description of what was just edited in the data cleaning."
            },
            {"role": "assistant", "content": f"The data cleaning has been updated to:\n{new_data_cleaning}"}
        ]
    )
    brief_description = description_response.choices[0].message['content'].strip()

    with st.chat_message("assistant"):
        st.markdown(f"Edit completed. {brief_description}")

# # # # # # # # # # # # # # 

def brainstorm_data_cleaning_edits(prompt, all_interpretations):
    """Help the user brainstorm possible data cleaning steps based on the dataset."""

    brainstorm_prompt = f"You are a data science assistant. The user is uncertain about what data cleaning to apply. " \
                        f"Based on the following dataset interpretations, suggest possible data cleaning steps:\n{all_interpretations}"

    brainstorm_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {"role": "system", "content": brainstorm_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    brainstorm_output = brainstorm_response.choices[0].message['content'].strip()

    with st.chat_message("assistant"):
        st.markdown(brainstorm_output)

# # # # # # # # # # # # # # 
# # # # # # # # # # # # # # 
# # # # # # # # # # # # # # 
# # # # # # # # # # # # # # 

def generate_feature_engineering_edits(column_interpretation, all_interpretations):
    """Generate Python code for feature engineering based on dataset interpretations."""

    # Initialize if not already done
    if 'all_feature_engineering' not in st.session_state:
        st.session_state.all_feature_engineering = ""

    # Check if feature engineering has already been generated
    if st.session_state.all_feature_engineering:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.all_feature_engineering)
        return
    
    user_prompt = f"Given the following dataset information:\n" \
                  f"Dataset Interpretation:\n{all_interpretations}\n" \
                  "Make sure to generate code for each type of feature engineering. " \
                  f"Column Interpretations:\n{column_interpretation}\n" \
                  "What potential feature engineering can you come up with? " \
                  "Make sure to generate code for each type of feature engineering. " \
                  "Print them in Python block codes."
    
    feature_engineering_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {"role": "system", "content": "You are a data science assistant specialized in feature engineering."},
            {"role": "user", "content": user_prompt}
        ]
    )
    feature_engineering_output = feature_engineering_response.choices[0].message['content'].strip()
    
    # Save the generated feature engineering to session state
    st.session_state.all_feature_engineering = feature_engineering_output

    with st.chat_message("assistant"):
        st.markdown(feature_engineering_output)

# # # # # # # # # # # # # # 

def manage_feature_engineering_edits(prompt):
    """Manage the editing of feature engineering."""

    if 'all_feature_engineering' not in st.session_state:
        st.session_state.all_feature_engineering = ""

    new_feature_engineering_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": f"Edit the following feature engineering based on the user's input:\n{st.session_state.all_feature_engineering}. Retain the existing output, do not delete them. Just add the edits"
            },
            {"role": "user", "content": prompt}
        ]
    )
    new_feature_engineering = new_feature_engineering_response.choices[0].message['content'].strip()
    
    st.session_state.all_feature_engineering = new_feature_engineering

    description_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": "Provide a brief description of what was just edited in the feature engineering."
            },
            {"role": "assistant", "content": f"The feature engineering has been updated to:\n{new_feature_engineering}"}
        ]
    )
    brief_description = description_response.choices[0].message['content'].strip()

    with st.chat_message("assistant"):
        st.markdown(f"Edit completed. {brief_description}")

# # # # # # # # # # # # # # 

def brainstorm_feature_engineering_edits(prompt, all_interpretations):
    """Help the user brainstorm possible feature engineering based on the dataset."""
    
    brainstorm_prompt = f"You are a data science assistant. The user is uncertain about what feature engineering to apply. " \
                        f"Based on the following dataset interpretations, suggest possible feature engineering:\n{all_interpretations}"
    
    brainstorm_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {"role": "system", "content": brainstorm_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    brainstorm_output = brainstorm_response.choices[0].message['content'].strip()
    
    with st.chat_message("assistant"):
        st.markdown(brainstorm_output)

# # # # # # # # # # # # # # 
# # # # # # # # # # # # # # 
# # # # # # # # # # # # # # 
# # # # # # # # # # # # # # 

def generate_machine_learning_model(column_interpretation, all_interpretations):
    """Generate Python code for a basic machine learning model based on dataset interpretations."""

    # Initialize if not already done
    if 'all_ml_models' not in st.session_state:
        st.session_state.all_ml_models = ""
    
    # Check if a model has already been generated
    if st.session_state.all_ml_models:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.all_ml_models)
        return
    
    user_prompt = f"Given the following dataset information:\n" \
                  f"Dataset Interpretation:\n{all_interpretations}\n" \
                  f"Column Interpretations:\n{column_interpretation}\n" \
                  "Create a basic machine learning model with steps. " \
                  "First, provide a brief general description of what the machine learning model might look like. " \
                  "Then, print the steps to build the model in Python block codes."
    
    ml_model_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {"role": "system", "content": "You are a data science assistant specialized in machine learning."},
            {"role": "user", "content": user_prompt}
        ]
    )
    ml_model_output = ml_model_response.choices[0].message['content'].strip()
    
    # Save the generated model to session state
    st.session_state.all_ml_models = ml_model_output
    
    with st.chat_message("assistant"):
        st.markdown(ml_model_output)


# # # # # # # # # # # # # # 

def manage_machine_learning_model(prompt):

    if 'all_ml_models' not in st.session_state:
        st.session_state.all_ml_models = ""
    
    new_ml_model_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": f"Edit the following machine learning model based on the user's input:\n{st.session_state.all_ml_models}"
            },
            {"role": "user", "content": prompt}
        ]
    )
    new_ml_model = new_ml_model_response.choices[0].message['content'].strip()
    
    # Update the model in session state
    st.session_state.all_ml_models = new_ml_model
    
    description_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {
                "role": "system",
                "content": "Provide a brief description of what was just edited in the machine learning model."
            },
            {"role": "assistant", "content": f"The machine learning model has been updated to:\n{new_ml_model}"}
        ]
    )
    brief_description = description_response.choices[0].message['content'].strip()

    with st.chat_message("assistant"):
        st.markdown(f"Edit completed. {brief_description}")


# # # # # # # # # # # # # # 

def brainstorm_machine_learning_model(prompt, all_interpretations):
    """Help the user brainstorm possible machine learning models based on the dataset."""
    
    # Initialize if not already done
    if 'all_ml_models' not in st.session_state:
        st.session_state.all_ml_models = ""
    
    existing_model_info = ""
    if st.session_state.all_ml_models:
        existing_model_info = f"So far, the following machine learning model has been generated:\n{st.session_state.all_ml_models}\n\n"
    
    brainstorm_prompt = f"You are a data science assistant. The user is uncertain about what machine learning model to apply. " \
                        f"{existing_model_info}" \
                        f"Based on the following dataset interpretations, suggest possible machine learning models:\n{all_interpretations}"
    
    brainstorm_response = openai.ChatCompletion.create(
        model=st.session_state["openai_model"],
        messages=[
            {"role": "system", "content": brainstorm_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    brainstorm_output = brainstorm_response.choices[0].message['content'].strip()
    
    with st.chat_message("assistant"):
        st.markdown(brainstorm_output)

# # # # # # # # # # # # # # 
# # # # # # # # # # # # # # 
# # # # # # # # # # # # # # 
# # # # # # # # # # # # # # 

# Main

image = st.image('logo.png')


# Initialize a variable to keep track of API key validity
if "is_valid_api_key" not in st.session_state:
    st.session_state["is_valid_api_key"] = False

# If the API key is not valid, show the input box
if not st.session_state["is_valid_api_key"]:
    api_key_input = st.text_input("Enter your OpenAI API key:", type="password")

    # Check the validity of the API key
    if api_key_input:
        try:
            openai.api_key = api_key_input
            # Make a test API call
            openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Who won the world series in 2020?"}]
            )
            st.session_state["is_valid_api_key"] = True
            st.success("Valid API key!")
        except Exception as e:
            st.error(f"Invalid API key: {e}")
            st.session_state["is_valid_api_key"] = False
else:
    st.write("API key is valid!")

# Only enable functionalities if the API key is valid
if st.session_state["is_valid_api_key"]:
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "dataset_interpretations" not in st.session_state:
        st.session_state.dataset_interpretations = {}

        # Initialize if not already done
    if 'all_ml_models' not in st.session_state:
        st.session_state.all_ml_models = ""

    if 'all_visualizations' not in st.session_state:
        st.session_state.all_visualizations = ""

    if 'all_feature_engineering' not in st.session_state:
        st.session_state.all_feature_engineering = ""

    if 'all_data_cleaning' not in st.session_state:
        st.session_state.all_data_cleaning = ""

    if 'prompt' not in st.session_state:
        st.session_state.prompt = ""

    if 'intent_content' not in st.session_state:
        st.session_state.intent_content = ""

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Dropdown for selecting a sample dataset
    sample_dataset = st.selectbox(
        'Select a sample dataset',
        ('None', 'Automobile.csv', 'countries-table.csv', 'ds_salaries.csv')
    )

    # Load the selected sample dataset
    if sample_dataset != 'None':
        data = pd.read_csv(f'{sample_dataset}')
        all_interpretations = interpret_csv(data)
        st.session_state.all_interpretations = all_interpretations
        st.session_state.dataset_interpreted = True

    # Handle CSV upload
    uploaded_file = st.file_uploader("Or upload your own CSV file", type=["csv"])

    if uploaded_file and "dataset_interpreted" not in st.session_state:
        data = pd.read_csv(uploaded_file)
        # Interpret the dataset immediately upon upload
        all_interpretations = interpret_csv(data)
        st.session_state.all_interpretations = all_interpretations  # Store in session state
        st.session_state.dataset_interpreted = True

    # Handle chat interface
    prompt = st.chat_input("Ask about the dataset:")

    response_content = ""

    # Inside your main Streamlit app logic
    if 'all_interpretations' in st.session_state:
        if prompt:
            # Existing code to identify user intent
            intent_response = openai.ChatCompletion.create(
                model='gpt-4',
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Identify the user's intent based on their input. The possible intents are:\n"
                            "- ASKING_ABOUT_DATASET: The user is inquiring about the dataset.\n"
                            "- EDITING_INTERPRETATION: The user wants to edit the dataset's interpretation.\n"
                            "- GENERATE_VISUALIZATION_QUESTIONS: The user wants to generate code blocks for visualizations.\n"
                            "- BRAINSTORMING_VISUALIZATION: The user is brainstorming about what visualizations to create.\n"
                            "- EDITING_VISUALIZATION: The user wants to edit the dataset's pre-made visualizations.\n"
                            "- GENERATE_FEATURE_ENGINEERING_EDITS: The user wants to generate code blocks for feature engineering.\n"
                            "- BRAINSTORMING_FEATURE_ENGINEERING: The user is brainstorming about what feature engineering to apply.\n"
                            "- EDITING_FEATURE_ENGINEERING: The user wants to edit the dataset's pre-made feature engineering.\n"
                            "- GENERATE_MACHINE_LEARNING_MODEL: The user wants to generate code blocks for creating machine learning models.\n"
                            "- BRAINSTORMING_MACHINE_LEARNING_MODEL: The user is brainstorming about what machine learning models to apply.\n"
                            "- EDIT_MACHINE_LEARNING_MODEL: The user wants to manage or edit the dataset machine learning models.\n"
                            "- GENERATE_DATA_CLEANING_EDITS: The user wants to generate code blocks for data cleaning.\n"
                            "- BRAINSTORMING_DATA_CLEANING: The user is brainstorming about what data cleaning to apply.\n"
                            "- EDITING_DATA_CLEANING: The user wants to edit the dataset's pre-made data cleaning."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            intent_content = intent_response.choices[0].message['content'].strip()

            if "ASKING_ABOUT_DATASET" in intent_content:
                # Consult the model to answer the question based on the dataset
                response = openai.ChatCompletion.create(
                    model=st.session_state["openai_model"],
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a helpful assistant. Answer the user's question based on the following dataset interpretations:\n{st.session_state.all_interpretations}"
                        },
                        {"role": "user", "content": prompt}
                    ]
                )
                response_content = response.choices[0].message['content'].strip()

            elif "EDITING_INTERPRETATION" in intent_content:
                manage_interpretations(prompt)

            elif "GENERATE_VISUALIZATION_QUESTIONS" in intent_content:
                generate_visualization_questions(st.session_state['all_interpretations'], st.session_state['all_interpretations'])

            elif "BRAINSTORMING_VISUALIZATION" in intent_content:
                brainstorm_visualizations(prompt, st.session_state['all_interpretations'])

            elif "EDITING_VISUALIZATION" in intent_content:
                manage_visualizations(prompt)

            elif "GENERATE_FEATURE_ENGINEERING_EDITS" in intent_content:
                generate_feature_engineering_edits(st.session_state['all_interpretations'], st.session_state['all_interpretations'])

            elif "BRAINSTORMING_FEATURE_ENGINEERING" in intent_content:
                brainstorm_feature_engineering_edits(prompt, st.session_state['all_interpretations'])

            elif "EDITING_FEATURE_ENGINEERING" in intent_content:
                manage_feature_engineering_edits(prompt)

            elif "GENERATE_MACHINE_LEARNING_MODEL" in intent_content:
                generate_machine_learning_model(st.session_state['all_interpretations'], st.session_state['all_interpretations'])

            elif "BRAINSTORMING_MACHINE_LEARNING_MODEL" in intent_content:
                brainstorm_machine_learning_model(prompt, st.session_state['all_interpretations'])

            elif "EDIT_MACHINE_LEARNING_MODEL" in intent_content:
                manage_machine_learning_model(prompt)

            elif "GENERATE_DATA_CLEANING_EDITS" in intent_content:
                generate_data_cleaning_edits(st.session_state['all_interpretations'], st.session_state['all_interpretations'])

            elif "BRAINSTORMING_DATA_CLEANING" in intent_content:
                brainstorm_data_cleaning_edits(prompt, st.session_state['all_interpretations'])

            elif "EDITING_DATA_CLEANING" in intent_content:
                manage_data_cleaning_edits(prompt)

            with st.chat_message("assistant"):
                st.markdown(response_content)