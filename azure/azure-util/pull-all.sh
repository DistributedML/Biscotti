

if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters (expecting 3):"
    echo "[ipfile, uname, branchname]"
    exit
fi

ipfile=$1
uname=$2
branchname=$3

bisPath="/home/$uname/gopath/src/Biscotti"
mnist_path="/home/$uname/gopath/src/Biscotti/ML/Pytorch/data/mnist"

hostFile="../azure-conf/$ipfile"

for ip in $(cat $hostFile);do

	ssh -t $uname@$ip "

		cd $bisPath
		git stash	
		git fetch origin
		git checkout $branchname	
		git pull origin $branchname
		cd $mnist_path
		python parse_mnist.py 200
	"

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
