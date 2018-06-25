timestamp() {
  date +"%T"
}


sudo iptables -A INPUT -p tcp --destination-port 8002 -j DROP
sudo iptables -A OUTPUT -p tcp --dport 8002 -j DROP

timestamp
# iptables save

sleep 30

timestamp

sudo iptables -F INPUT
sudo iptables -F OUTPUT


