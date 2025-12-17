from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


def login_with_service_account():
    """
    Google Drive service with a service account.
    note: for the service account to work, you need to share the folder or
    files with the service account email.

    :return: google auth
    """
    # Define the settings dict to use a service account
    # We also can use all options available for the settings dict like
    # oauth_scope,save_credentials,etc.
    settings = {
                "client_config_backend": "service",
                "service_config": {
                    "client_json_file_path": "cj_service_key.json",
                }
            }
    # Create instance of GoogleAuth
    gauth = GoogleAuth(settings=settings)
    # Authenticate
    gauth.ServiceAuth()
    return gauth

try:
    gauth = GoogleAuth()
    gauth.ServiceAuth()
    print("Authentication successful!")
except Exception as e:
    print(f"Error: {e}")
drive=GoogleDrive(gauth)

folder_id='1y-VGoIZJs6WX9xGkPTB7hkRUYyhNUxZd'

try:
    # Try to get folder metadata
    folder = drive.CreateFile({'id': folder_id})
    folder.FetchMetadata()
    print(f"Folder title: {folder['title']}")
    print(f"Permissions: {folder.get('permissions', 'No permissions data')}")
    
    # List files in folder
    file_list = drive.ListFile({
        'q': f"'{folder_id}' in parents and trashed=false"
    }).GetList()
    
    print(f"Found {len(file_list)} files in folder")
    
except Exception as e:
    print(f"Error accessing folder: {e}")

file_list = drive.ListFile().GetList()
for file1 in file_list:
    print('title: %s, id: %s' % (file1['title'], file1['id']))
