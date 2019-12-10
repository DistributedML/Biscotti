azuredeployScript="$GOPATH/src/Biscotti/azure/azure-run"
logFiles="$GOPATH/src/Biscotti/DistSys/LogFiles"

filename="epsilon_05.log"
cd $azuredeployScript
bash runBiscotti.sh 5 100 hosts_diffDC mnist '-ns=70 -ep=0.5 -np=false'
cd $logFiles
mv log_0_100.log ../../eval_privacy_utility_krum/$filename	