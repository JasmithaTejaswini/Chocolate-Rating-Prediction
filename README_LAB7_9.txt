Chocolate Bar Rating - Lab 7 to Lab 9 source scripts

Copy the src folder from this archive into the root of your existing 241FA18484 project.
If PowerShell asks whether to merge folders, choose Yes. Do not replace your existing
src/preprocess.py, src/train.py, or any existing Lab 3-6 files.

Run from the project root:
  python .\src\mlops_pipeline\01_load_data.py
  python .\src\mlops_pipeline\02_validate_data.py
  python .\src\mlops_pipeline\03_preprocessing.py
  python .\src\mlops_pipeline\04_validate_outputs.py
  python .\src\mlops_pipeline\05_train_registry.py
  python .\src\mlops_pipeline\06_model_evaluation.py
  python .\src\mlops_pipeline\07_quality_gate.py
  python .\src\mlops_pipeline\08_automate_lifecycle.py
  python .\src\mlops_pipeline\09_inference.py
  python .\src\mlops_pipeline\10_failure_injection.py

Or run the sequence:
  python .\src\mlops_pipeline\chocolate_rating_pipeline.py

Important:
- These scripts are an adaptation based on the Lab 7-9 Churn project structure you supplied.
  They are not verified against an official Chocolate project Lab 7-9 specification.
- Lab 7-9 pipeline assumes the existing project files:
  src/preprocess.py, src/train.py
  data/raw/flavors_of_cacao.csv
- Lab 9 failure simulations run in memory and do not overwrite the original CSV.
- The MLflow script logs the selected model artifact and run metadata. It does not register
  a model version in the MLflow Model Registry.
- Review the scripts and test them in your virtual environment before relying on the outputs.
