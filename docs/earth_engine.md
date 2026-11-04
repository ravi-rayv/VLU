# Earth Engine access

Building heights come from the **Open Buildings 2.5D Temporal** dataset in Google Earth Engine (`GOOGLE/Research/open-buildings-temporal/v1`). Each user needs their own Earth Engine project. It is free for non-commercial and research use.

## 1. Register a project (once)

1. Go to <https://code.earthengine.google.com/register>.
2. Choose *Use with a Cloud project* and either create a new project or select an existing one.
3. Choose *Unpaid usage* and *Academia & research* (or the category that fits you).
4. Note the **project ID**, e.g. `my-ee-project-123`.

## 2a. On your own computer (recommended)

```bash
pip install earthengine-api
earthengine authenticate          # opens a browser once; credentials are saved locally
```

Then enter the project ID in the portal sidebar (*This computer*) or pass `--ee-project my-ee-project-123` on the command line.

## 2b. On a hosted portal (Streamlit Cloud, Hugging Face, a lab server)

A hosted app cannot open your browser for sign-in. Use your **own** service account instead:

1. In the Google Cloud Console for your project, open *IAM & Admin → Service accounts* and create a service account.
2. Give it the role **Earth Engine Resource Viewer** (or *Service Usage Consumer* + *Earth Engine Resource Viewer*).
3. Register the service account for Earth Engine: <https://signup.earthengine.google.com/#!/service_accounts>.
4. Create a **JSON key** for it and download the file.
5. In the portal choose *Upload my own service-account key (JSON)*, upload the file and enter the project ID.

The key is read into memory for your session only. It is never written to disk or shared, and runs use your own project's quota.

Delete the key in the Cloud Console when you no longer need it.
