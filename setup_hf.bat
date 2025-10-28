@echo off
echo 🚀 Setting up Hugging Face Authentication for XAUUSD Dataset Upload
echo ====================================================================
echo.
echo To upload your XAUUSD dataset to Hugging Face, follow these steps:
echo.
echo 1. 📝 Create a Hugging Face account (if you don't have one):
echo    Go to: https://huggingface.co/join
echo.
echo 2. 🔑 Get your API token:
echo    Go to: https://huggingface.co/settings/tokens
echo    Click "New token"
echo    Name it "XAUUSD Dataset Upload"
echo    Set type to "Write"
echo    Copy the token
echo.
echo 3. 🔐 Set your token as environment variable:
echo    (Replace YOUR_TOKEN_HERE with your actual token)
echo.
set /p HF_TOKEN="Enter your Hugging Face token: "
echo.
echo ✅ Token set! Now running authentication setup...
echo.
python setup_hf_auth.py
echo.
echo 🎯 If authentication was successful, run the next command:
echo python upload_datasets.py
echo.
pause