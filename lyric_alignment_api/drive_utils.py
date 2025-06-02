from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SERVICE_ACCOUNT_FILE = '/content/drive/MyDrive/micro-answer-461617-j3-1cf2f152d197.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def upload_json_to_drive(file_path, filename, folder_id=None):
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )

    service = build('drive', 'v3', credentials=creds)

    file_metadata = {
        'name': filename,
        'mimeType': 'application/json'
    }

    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(file_path, mimetype='application/json')
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    # Set permission: anyone with link can view
    service.permissions().create(
        fileId=file['id'],
        body={'role': 'reader', 'type': 'anyone'},
    ).execute()

    # Return public link
    return f"https://drive.google.com/uc?id={file['id']}&export=download"
