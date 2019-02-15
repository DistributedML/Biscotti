azuredeployScript="$GOPATH/src/Biscotti/azure/azure-run"
logFiles="$GOPATH/src/Biscotti/DistSys/LogFiles"

filename="3_3_5.log"
cd $azuredeployScript
bash runBiscotti.sh 5 100 hosts_diffDC mnist '-nn=3 -nv=3 -na=5'
cd $logFiles
mv log_0_100.log ../../eval_vrf/$filename