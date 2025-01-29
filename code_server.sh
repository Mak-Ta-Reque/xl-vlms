#!/bin/bash
# Replace <PASSWORD HASH> below with the argon2 hash of the
# password of your choice.
# Run the following command and copy the output here:
#
# echo -n <your password> | argon2 $(openssl rand -base64 32) -e
#
export PASSWORD=gfMVeMhwll
# Choose a port based on the job id
export PORT=$(((${SLURM_JOB_ID} + 10007) % 16384 + 49152))

# Use the latest version of code server
export CODE_SERVER="$(find /netscratch/software/ -name 'code-server-*-linux-amd64.tar.gz' | sort | tail -1)"
if [ -z "$CODE_SERVER" ]
then
      echo "ERROR: no code server package found; check that /netscratch/software is in --container-mounts"
      exit 1
fi

# Print the URL where the IDE will become available
echo
echo =========================================
echo =========================================
echo =========================================
echo
echo using $CODE_SERVER
echo
echo IDE will be available at:
echo
echo $HOSTNAME.kl.dfki.de:$PORT
echo
echo Please wait for setup to finish.
echo
echo =========================================
echo =========================================
echo =========================================
echo

# Extract the IDE files
tar -f "$CODE_SERVER" -C /tmp/ -xz

# Install extensions
/tmp/code-server-*/bin/code-server \
    --user-data-dir=.code-server \
    --install-extension="ms-python.python" \
    # --install-extension="ms-python.vscode-pylance" \

# Start the IDE
/tmp/code-server-*/bin/code-server \
    --disable-telemetry \
    --disable-update-check \
    --bind-addr=$HOSTNAME.kl.dfki.de:$PORT \
    --auth password \
    --cert \
    --cert-host=$HOSTNAME.kl.dfki.de \
    --user-data-dir=.code-server \
    "$(pwd)"

