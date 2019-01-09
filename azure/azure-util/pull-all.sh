

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters (expecting 2):"
    echo "[ipfile]"
    exit
fi

uname=shayan
ipfile=$1

hostFile="../azure-conf/$ipfile"

for ip in $(cat $hostFile);do

	ssh -t $uname@$ip "
		cd $GOPATH/src/Biscotti
		git remote set-url origin https://github.com/DistributedML/Biscotti.git
		git stash
		git pull origin master
	"
	# break
done

exit
