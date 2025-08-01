
"""
import subprocess



# names
output = subprocess.check_output(
        "pip list | tail -n +3 | awk '{print $1}'",
        shell=True,
        text=True  # ensures output is returned as a string
        )
package_names = output.strip().split('\n')
print(type(package_names))
print(package_names)
print(type(package_names))


# versions
output = subprocess.check_output(
        "pip list | tail -n +3 | awk '{print $2}'",
        shell=True,
        text=True  # ensures output is returned as a string
        )
versions = output.strip().split('\n')
print(type(versions))
print(versions)
print(type(versions))


# available versions
packages_list=[]
for package, current_version in zip(package_names, versions):

	# get latest version for a given package
	command=(f"pip index versions {package}")
        result = subprocess.check_output(command, shell=True, text=True)
	latest_version=result.split(":")[1].split(",")[0].lstrip() #first version is latest

	# store information in output list
	packages_list.append({'name':package,'current':current_version,'latest':latest_version})

	return packages_list
"""

import subprocess
import json

# Get current installed packages
output = subprocess.check_output(
    ["pip", "list", "--format=json"],
    text=True
)
packages = json.loads(output)
print(packages)


packages_list = []

for pkg in packages:
    name = pkg['name']
    current_version = pkg['version']
    
    try:
        # Get latest version using pip index
        result = subprocess.check_output(
            f"pip index versions {name}",
            shell=True,
            text=True
        )
        latest_version = result.split(":")[1].split(",")[0].strip()
    except Exception as e:
        latest_version = "unknown"

    packages_list.append({
        'name': name,
        'current': current_version,
        'latest': latest_version
    })

print(packages_list)
