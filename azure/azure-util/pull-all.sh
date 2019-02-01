

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters (expecting 2):"
    echo "[ipfile]"
    exit
fi

uname="matheus"
ipfile=$1
bisPath="/home/$uname/gopath/src/Biscotti"


hostFile="../azure-conf/$ipfile"

for ip in $(cat $hostFile);do

	ssh -t $uname@$ip "
		cd $bisPath
		# git remote set-url origin https://github.com/DistributedML/Biscotti.git
		# git stash
		git pull origin krum-implementation
		# cd /home/matheus/gopath/src/Biscotti/ML/Pytorch/data/mnist
		# python parse_mnist.py
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
