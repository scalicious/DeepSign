# DeepSign

## Run the image enhancement UI

The UI and enhancement endpoint are served from one local Python process. The
enhancement endpoint uses the existing implementation in `enhance.py` and does
not change or load Model F.

1. Install Python 3.10 or newer.
2. From this folder, install the enhancement dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

3. Start the local UI and enhancement service:

   ```powershell
   py server.py
   ```

4. Open [http://127.0.0.1:8765/](http://127.0.0.1:8765/) and upload an image.

The before/after comparison appears once enhancement completes. The local
service accepts image uploads up to 20 MB and listens only on `127.0.0.1` by
default. Stop it with Ctrl+C.
