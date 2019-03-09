# Run with number of noisers, verifiers and aggregators
# Dataset and deployment filename harcoded

if [ "$#" -ne 3 ]; then

    echo "Illegal number of parameters"
    echo "usage:"
    echo "[numNoisers, numVerifiers, numAggregators]"
    exit

fi

azuredeployScript="$GOPATH/src/Biscotti/azure/azure-run"
logFiles="$GOPATH/src/Biscotti/DistSys/LogFiles"

numNoisers=$1
numVerifiers=$2
numAggregators=$3

filename=$numNoisers"_"$numVerifiers"_"5.log
echo $filename

cd $azuredeployScript
bash runBiscotti.sh 5 100 hosts_diffDC mnist '-nn=3 -nv=3 -na=5'
mv log_0_100.log ../../eval_vrf/$filename