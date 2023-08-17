import subprocess
import time

# List of script filenames to execute
scripts_to_run = ['/Users/arnau/Desktop/stock_website/financial-stocks-website/creat_table_financial.py', 'import_financial.py']

# Time to sleep between script executions (in seconds)
sleep_time = 10

# Iterate over the list and execute each script
for script in scripts_to_run:
    subprocess.run(['python3', script])
    time.sleep(sleep_time)

