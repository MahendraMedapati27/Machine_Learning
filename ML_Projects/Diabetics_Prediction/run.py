import re
import sys
from streamlit.web import cli as stcli

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", r"C:\Users\Hp\OneDrive\Desktop\ML\Diabetics_Prediction\App.py"]  # Replace 'your_script.py' with the path to your Streamlit script
    stcli.main()
