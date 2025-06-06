@echo off
REM Run from project root. Generates /dist/app.exe
pyinstaller ^
    --onefile ^
    --add-data "models/model.joblib;models" ^
    --add-data "assets/example_input.xlsx;assets" ^
    --collect-submodules "sklearn" ^
    src/app.py
pause
