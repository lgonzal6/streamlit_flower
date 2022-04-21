
# Streamlit 

Streamlit is an opinionated library for serving models and visualizations in pure Python. It is interactive and quick to deploy.

## Learning objectives

- Create a streamlit app to serve models and visualizations
- Use plotly express to create interactive visualizations
- Deploy your streamlit on streamlit

## Install packages

In your terminal, list the packages in your virtual environments.

If you don't see streamlit and plotly, then install them.

`pip install streamlit`
`pip install plotly`

## Create a Streamlit app

1. Arrange your screen(s) so you can see a browser window and your text editor. 
2. Open your text editor
3. Navigate to the folder where you want to create your app.
4. Make a new python file named `my_app.py`.
5. In the file, add `import streamlit as st` and save the file.
6. In the terminal, in the same directory as your `my_app.py` file, execute `streamlit run my_app.py`.
    - You are now serving your app locally.
    - You should see a browser window pop up.
    - After changing your code, a button in the top right of the window can make it auto-reload.
    - In Streamlit, your code runs from top to bottom, every time a user does something.
    - You can deploy your model to the internet by pushing it to your personal GitHub account and connecting that repo to Streamlit's servers.
7. If you want to deploy your app to the web on Streamlit the company's servers, you will need a free Community account. You can sign up [here](https://streamlit.io/cloud). Push your repo to GitHub with a requirements.txt file and follow the instructions [here](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app).

