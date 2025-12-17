from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.ServiceAuth()
drive = GoogleDrive(gauth)
# gauth.LoadCredentialsFile("credentials_module.txt")

# if gauth.credentials is None:
#     gauth.LocalWebserverAuth()  # Only first time
# elif gauth.access_token_expired:
#     gauth.Refresh()  # Silent refresh
# else:
#     gauth.Authorize()  # Silent authorize

# gauth.SaveCredentialsFile("credentials_module.txt")
# drive = GoogleDrive(gauth)
