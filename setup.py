import subprocess, time

process = subprocess.call("pip3 install -r requirements.txt", shell=True)

if process:
    print("JSONIC kuruldu!")
    time.sleep(2)
    print("Bu pncere 10 saniye içinde kapanacak...!")
    time.sleep(10)