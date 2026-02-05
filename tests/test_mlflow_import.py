def test_mlflow_import_smoke():
    import mlflow

    assert mlflow.__version__
