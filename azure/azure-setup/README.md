These scripts assume that you have the azure CLI installed:
https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest

The command to log you in is:
az login

After that follow the procedure below to setup VM's for Biscotti.

1. Set up a resource group. All VM's part of this group. Run the following script
	
	bash create-rg.sh name location

	Ex: bash create-rg.sh biscotti eastus

2. Set up virtual networks. Virtual Network setup will be different based on whether you are deploying in same location or different locations. Different locations will have different vnets and peering between these networks need to be set up.  For same location, one vnet needs to be set up. Run the following script:

	bash create-vnets.sh $rgroup $location

	For same location deployment:
		bash create-vnets.sh biscotti same
	For different:
		bash create-vnets.sh biscotti diff

	Locations to deploy hardcoded in script. Change if needed.


2. Set up the number of VM's that you want in that resource group. Run the following script:

	bash azure-deploy-vms.sh numVMs rgroup user password

	Ex: bash azure-deploy-vms.sh 20 rgroup shayan password diff

	Note: There is some stuff hardcoded in the script that you might want to change.
	1. List of locations -> VM's evenly spread out in these locations
	2. Your ssh key path.
	3. vmprefix -> VM's created with name bis0, bis1 etc 
	4. vm image -> UBUNTULTS

3. Get the ips of the vms in a text file created by running the following script:
	
	bash get-vm-ips.sh rgroup numVMs ipFileName

	Ex: bash get-vm-ips.sh biscotti 20 hosts_diffDC

	Note: Hardcoded path for output file in script -> ../azure-conf/

4. Run an install script on each VM. Sets up Biscotti repo, installs packages/dependencies.

	bash set-up-biscotti-all-vms.sh uname ipfilename installscript		

	Note: Install script should be in same folder as the above scripts.

5. Generate an ssh key on each VM and copy over to your machine. This will enable smooth transfer of logfiles to your machine after biscotti-run.

	bash gen-vm-keys.sh uname ipfilename

	Ex: bash gen-vm-keys.sh shayan hosts_diffDC

	Note: Hardcoded path for output file in script -> ../azure-conf/

	bash get-vm-keys.sh localuname localip pass ipfile

	Ex: bash get-vm-keys.sh shayan 198.162.52.126 pass hosts_diffDC

	localuname -> username of your local machine
	ip -> public ip of your local machine
	pass -> your local login pass	
