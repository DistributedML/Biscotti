

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters (expecting 2):"
    echo "[ipfile]"
    exit
fi

uname="shayan"
ipfile=$1
bisPath="/home/$uname/gopath/src/Biscotti"


hostFile="../azure-conf/$ipfile"

for ip in $(cat $hostFile);do

	ssh -t $uname@$ip "
		cd $bisPath
		git fetch origin
		git checkout krum_eval_rs
		git pull origin krum_eval_rs
	"
	# break
done

exit

# Switch branch
	# git fetch origin
	# git checkout krum-implementation
	# git pull origin krum-implementation

# Set remote url
# git remote set-url origin https://github.com/DistributedML/Biscotti.git
	# cd /home/$uname/gopath/src/Biscotti/ML/Pytorch/data/mnist
		# python parse_mnist.py
