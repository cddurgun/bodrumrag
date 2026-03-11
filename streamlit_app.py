"""
Entry point for Streamlit Community Cloud.
Streamlit Cloud looks for streamlit_app.py by default.
"""
import runpy
runpy.run_module("src.app", run_name="__main__")
