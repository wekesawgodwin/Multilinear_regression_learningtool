"""
Data loading module for the Regression Explorer.
Uses a factory pattern to handle CSV, Excel, and ZIP files.
"""

import pandas as pd
import zipfile
from abc import ABC, abstractmethod


class DataLoader(ABC):
    """Abstract base class for file loaders."""
    @abstractmethod
    def load(self, file) -> pd.DataFrame:
        pass


class CSVLoader(DataLoader):
    def load(self, file) -> pd.DataFrame:
        return pd.read_csv(file)


class ExcelLoader(DataLoader):
    def load(self, file) -> pd.DataFrame:
        return pd.read_excel(file, engine='openpyxl')


class DataLoaderFactory:
    """Factory that returns the correct loader based on file extension."""
    @staticmethod
    def get_loader(file_extension: str) -> DataLoader:
        ext = file_extension.lower()
        if ext in ['.csv', '.txt']:
            return CSVLoader()
        elif ext in ['.xlsx', '.xls']:
            return ExcelLoader()
        else:
            raise ValueError(f"Unsupported file type: {ext}")


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Load an uploaded file (CSV, Excel, or ZIP).
    For ZIP archives, the user selects one inner file.
    """
    import streamlit as st  # avoid top-level import to keep module UI-agnostic? We can import here for selectbox.
    if uploaded_file.name.endswith('.zip'):
        with zipfile.ZipFile(uploaded_file) as zf:
            inner_files = [f for f in zf.namelist() if not f.endswith('/')]
            if not inner_files:
                st.error("ZIP archive contains no files.")
                return None
            selected_inner = st.selectbox("Select a file from the ZIP archive:", inner_files)
            with zf.open(selected_inner) as inner_file:
                inner_ext = '.' + selected_inner.rsplit('.', 1)[-1] if '.' in selected_inner else '.csv'
                loader = DataLoaderFactory.get_loader(inner_ext)
                return loader.load(inner_file)
    else:
        ext = '.' + uploaded_file.name.rsplit('.', 1)[-1] if '.' in uploaded_file.name else '.csv'
        loader = DataLoaderFactory.get_loader(ext)
        return loader.load(uploaded_file)